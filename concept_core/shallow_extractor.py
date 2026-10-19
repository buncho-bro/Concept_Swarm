"""
shallow_extractor.py -- 全大円干渉型 概念核抽出器
=====================================================
(Great-Circle Interference Concept-Core Extractor, CPU only / 0 MB VRAM)

パイプライン
------------
1. ShallowWeights   : 埋め込み層 (+ 第0層 input_layernorm の重み) だけを読む軽量ローダー。
                      ローカルキャッシュに shard があれば mmap で該当テンソルのみ参照し、
                      無ければ HTTP Range リクエストで該当テンソルのバイト列だけを取得する
                      (巨大な shard ファイル全体はダウンロードしない)。
2. GreatCircleField : 全トークンの単位ベクトル v_i と文脈進行方向 u_i から大円
                      G_i = span(v_i, u_i) ∩ S^{D-1} を張り、球面干渉場
                          Phi(x) = sum_i exp( -theta_i(x)^2 / (2 sigma^2) )
                      を構築する。高次元では法線が一意に定まらないため、
                          cos theta_i(x) = || P_i x ||   (P_i: 2次元平面への直交射影)
                          sin theta_i(x) = || x - P_i x || (直交補空間成分)
                      として厳密に距離を計算する。
3. 概念核抽出        : 全大円ペアの最近接点 (主角 SVD) を初期候補とし、
                      球面上のリーマン勾配上昇で干渉極大点 (焦点) に収束させ、
                      角度的に分離した上位 K 個を多重概念核として出力する。

設計上の制約
------------
* 全トークンを対等に扱う (頻度・IDF 等による重み付けや足切りはしない)。
* 既存プロトタイプへのスナップはしない。出力は干渉場の極大点そのもの。
* 全演算は NumPy (CPU)。torch は safetensors の CPU 読み込みにのみ使用し、
  CUDA API は一切呼ばない。
  ※ このモジュールは import 時に CUDA_VISIBLE_DEVICES を書き換えない。
    (GPU で生成を行う別プロセス、例えば Project Ripple に import された場合に
     GPU を巻き添えで無効化しないため。GPU を完全遮断したい実行スクリプト側で設定すること)
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"

EMBED_KEY_CANDIDATES = (
    "model.embed_tokens.weight",                      # Llama / Qwen / Mistral 系
    "transformer.embedding.word_embeddings.weight",   # ChatGLM / GLM 系
    "model.embed_tokens",                             # 一部の変種
    "transformer.wte.weight",                         # GPT-2 系
    "embeddings.word_embeddings.weight",              # BERT 系
)
LAYER0_NORM_CANDIDATES = (
    "model.layers.0.input_layernorm.weight",
    "transformer.encoder.layers.0.input_layernorm.weight",
    "transformer.h.0.ln_1.weight",
)

_DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}


def _default_cache_root() -> Path:
    return Path(os.environ.get("SHALLOW_EXTRACTOR_CACHE",
                               Path.home() / ".cache" / "shallow_extractor"))


# =============================================================================
# 1. Shallow weight loader
# =============================================================================

class ShallowWeights:
    """埋め込み行列と第0層 input_layernorm だけにアクセスするローダー。

    埋め込み行列全体をメモリに展開せず、必要なトークン行だけを
    safetensors の slice 読み込み (mmap) で取り出す。
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        use_layer0_norm: bool = True,
        center: bool = True,
        cache_root: Optional[Path] = None,
        verbose: bool = True,
    ) -> None:
        self.model_id = model_id
        self.use_layer0_norm = use_layer0_norm
        self.center = center
        self.verbose = verbose
        self.cache_dir = Path(cache_root or _default_cache_root()) / model_id.replace("/", "--")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.source = ""            # "local-shard" | "range-extract"
        self.bytes_fetched = 0      # 今回ネットワークから取得したバイト数
        self.tensor_bytes = 0       # 参照対象テンソルの合計バイト数 (= 実フットプリント)

        self._file, self.embed_key, self.norm_key = self._resolve()
        self.vocab_rows, self.dim = self._embed_shape()
        self._norm_w: Optional[np.ndarray] = self._load_norm() if use_layer0_norm else None
        self._mu: Optional[np.ndarray] = self._vocab_mean() if center else None
        self._row_cache: Dict[int, np.ndarray] = {}

    # -- file resolution -----------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[ShallowWeights] {msg}")

    def _weight_map(self) -> Dict[str, str]:
        from huggingface_hub import hf_hub_download
        try:
            idx = hf_hub_download(self.model_id, "model.safetensors.index.json")
            with open(idx, "r", encoding="utf-8") as f:
                return json.load(f)["weight_map"]
        except Exception:
            pass
        # 単一ファイル構成: ローカルにあればそのヘッダ、無ければリモートのヘッダだけ読む
        try:
            from safetensors import safe_open
            local = hf_hub_download(self.model_id, "model.safetensors", local_files_only=True)
            with safe_open(local, framework="pt", device="cpu") as f:
                return {k: "model.safetensors" for k in f.keys()}
        except Exception:
            return {k: "model.safetensors" for k in self._remote_header("model.safetensors")[0]}

    @staticmethod
    def _pick(keys: Sequence[str], candidates: Sequence[str], fuzzy: Tuple[str, ...]) -> Optional[str]:
        for c in candidates:
            if c in keys:
                return c
        for k in keys:
            if any(s in k for s in fuzzy) and k.endswith("weight"):
                return k
        return None

    def _resolve(self) -> Tuple[Path, str, Optional[str]]:
        extract = self.cache_dir / "shallow.safetensors"
        meta = self.cache_dir / "shallow_meta.json"
        if extract.exists() and meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            self.source = "range-extract (cached)"
            self.tensor_bytes = extract.stat().st_size
            self._log(f"using cached shallow extract: {extract}")
            return extract, m["embed_key"], m.get("norm_key")

        wmap = self._weight_map()
        keys = list(wmap.keys())
        embed_key = self._pick(keys, EMBED_KEY_CANDIDATES, ("embed_tokens", "word_embeddings", "wte"))
        if embed_key is None:
            raise KeyError(f"embedding tensor not found in {self.model_id}")
        norm_key = self._pick(keys, LAYER0_NORM_CANDIDATES, ("layers.0.input_layernorm",))
        needed = [k for k in (embed_key, norm_key) if k]
        shards = sorted({wmap[k] for k in needed})

        # (a) 既にローカルキャッシュに shard がある場合: そのまま mmap 参照 (ダウンロード 0)
        from huggingface_hub import hf_hub_download
        if len(shards) == 1:
            try:
                local = hf_hub_download(self.model_id, shards[0], local_files_only=True)
                self.source = "local-shard (mmap)"
                self._log(f"local shard found, reading only {needed} via mmap: {local}")
                return Path(local), embed_key, norm_key
            except Exception:
                pass

        # (b) 無い場合: HTTP Range で該当テンソルのバイト列だけ取得し、小さな safetensors を作る
        self._log(f"range-downloading only {needed} from {shards} (no full shard download)")
        tensors = {}
        for shard in shards:
            want = [k for k in needed if wmap[k] == shard]
            tensors.update(self._range_fetch(shard, want))
        _write_safetensors(extract, tensors)
        meta.write_text(json.dumps({"embed_key": embed_key, "norm_key": norm_key,
                                    "model_id": self.model_id}), encoding="utf-8")
        self.source = "range-extract"
        self.tensor_bytes = extract.stat().st_size
        return extract, embed_key, norm_key

    def _remote_header(self, shard: str):
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        with fs.open(f"{self.model_id}/{shard}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        return header, n

    def _range_fetch(self, shard: str, keys: Sequence[str]):
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        out = {}
        with fs.open(f"{self.model_id}/{shard}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            for k in keys:
                info = header[k]
                b, e = info["data_offsets"]
                f.seek(8 + n + b)
                buf = f.read(e - b)
                self.bytes_fetched += len(buf)
                out[k] = (info["dtype"], info["shape"], buf)
                self._log(f"  fetched {k} {info['shape']} {info['dtype']} ({len(buf)/1e6:.1f} MB)")
        return out

    # -- tensor access ---------------------------------------------------------

    def _open(self):
        from safetensors import safe_open
        return safe_open(str(self._file), framework="pt", device="cpu")

    def _embed_shape(self) -> Tuple[int, int]:
        with self._open() as f:
            shape = f.get_slice(self.embed_key).get_shape()
            if not self.tensor_bytes:
                # local shard の場合: 参照するテンソル分だけをフットプリントとして数える
                dt = f.get_slice(self.embed_key).get_dtype()
                self.tensor_bytes = int(np.prod(shape)) * _DTYPE_BYTES.get(dt, 2)
        return int(shape[0]), int(shape[1])

    def _load_norm(self) -> Optional[np.ndarray]:
        if not self.norm_key:
            self._log("layer-0 input_layernorm not found; using raw embeddings")
            return None
        with self._open() as f:
            w = f.get_tensor(self.norm_key).float().numpy().astype(np.float64)
        return w

    def _transform(self, rows: np.ndarray) -> np.ndarray:
        """第0層が実際に受け取る向き (RMSNorm の要素ごとスケール) に揃える。
        RMSNorm の 1/rms 部分は後段の L2 正規化で消えるので重みの乗算のみで等価。"""
        return rows * self._norm_w if self._norm_w is not None else rows

    def _vocab_mean(self, chunk: int = 8192) -> np.ndarray:
        """語彙全体の平均ベクトル (異方性除去用)。1 回だけ計算してキャッシュする。"""
        tag = "norm" if self._norm_w is not None else "raw"
        path = self.cache_dir / f"vocab_mean_{tag}.npy"
        if path.exists():
            return np.load(path)
        self._log("computing vocabulary mean (one-time, chunked, CPU)...")
        t0 = time.perf_counter()
        acc = np.zeros(self.dim, dtype=np.float64)
        with self._open() as f:
            sl = f.get_slice(self.embed_key)
            for s in range(0, self.vocab_rows, chunk):
                rows = sl[s:min(s + chunk, self.vocab_rows)].float().numpy().astype(np.float64)
                acc += self._transform(rows).sum(0)
        mu = acc / self.vocab_rows
        np.save(path, mu)
        self._log(f"vocabulary mean cached ({time.perf_counter() - t0:.1f}s): {path}")
        return mu

    def raw_rows(self, ids: Sequence[int]) -> np.ndarray:
        """生の埋め込み行 (float64)。必要な行だけを slice 読み込みする。"""
        missing = sorted({int(i) for i in ids if int(i) not in self._row_cache})
        if missing:
            with self._open() as f:
                sl = f.get_slice(self.embed_key)
                for i in missing:
                    self._row_cache[i] = sl[i:i + 1].float().numpy()[0].astype(np.float64)
        return np.stack([self._row_cache[int(i)] for i in ids]) if len(ids) else np.zeros((0, self.dim))

    def unit_vectors(self, ids: Sequence[int]) -> np.ndarray:
        """前処理済み (第0層ノルム重み -> 語彙平均除去 -> L2 正規化) の単位ベクトル。"""
        X = self._transform(self.raw_rows(ids))
        if self._mu is not None:
            X = X - self._mu
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, 1e-12)

    def preprocess_block(self, rows: np.ndarray) -> np.ndarray:
        """外部で読み込んだ埋め込み行ブロックに同じ前処理を適用する (評価用)。"""
        X = self._transform(rows.astype(np.float64))
        if self._mu is not None:
            X = X - self._mu
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    def iter_vocab_blocks(self, chunk: int = 8192, limit: Optional[int] = None):
        """語彙行列をブロック単位で読み出す (評価用: 語彙近傍検索など)。"""
        n = min(self.vocab_rows, limit or self.vocab_rows)
        with self._open() as f:
            sl = f.get_slice(self.embed_key)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                yield s, sl[s:e].float().numpy()


def _write_safetensors(path: Path, tensors: Dict[str, Tuple[str, list, bytes]]) -> None:
    header, offset = {}, 0
    for name, (dtype, shape, buf) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(buf)]}
        offset += len(buf)
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, (_, _, buf) in tensors.items():
            f.write(buf)
    os.replace(tmp, path)


# =============================================================================
# 2. Great-circle geometry & spherical interference field
# =============================================================================

def _normalize_rows(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


class GreatCircleField:
    """N 本の大円 G_i = span(v_i, u_i) ∩ S^{D-1} が作る球面干渉場。"""

    def __init__(self, V: np.ndarray, seed: int = 0) -> None:
        if V.ndim != 2 or len(V) < 2:
            raise ValueError("GreatCircleField needs at least 2 token vectors")
        self.V = _normalize_rows(V.astype(np.float64))
        self.U = self._context_directions(self.V, np.random.default_rng(seed))

    @staticmethod
    def _context_directions(V: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """局所的な文脈進行方向 d_i (中心差分、端点は片側差分) を v_i の接空間へ射影。"""
        N = len(V)
        D = np.empty_like(V)
        D[0] = V[1] - V[0]
        D[-1] = V[-1] - V[-2]
        if N > 2:
            D[1:-1] = V[2:] - V[:-2]
        D -= np.sum(D * V, axis=1, keepdims=True) * V
        norms = np.linalg.norm(D, axis=1)
        bad = norms < 1e-8
        if bad.any():
            # 同一トークンの反復などで方向が退化した場合: 系列重心方向 -> それも退化ならランダム直交方向
            c = V.mean(0)
            for i in np.where(bad)[0]:
                d = c - (c @ V[i]) * V[i]
                if np.linalg.norm(d) < 1e-8:
                    d = rng.standard_normal(V.shape[1])
                    d -= (d @ V[i]) * V[i]
                D[i] = d
            norms = np.linalg.norm(D, axis=1)
        return D / norms[:, None]

    # -- field evaluation ------------------------------------------------------

    def _angles(self, X: np.ndarray):
        a = X @ self.V.T                     # (M, N)  x·v_i
        b = X @ self.U.T                     # (M, N)  x·u_i
        r = np.clip(np.sqrt(a * a + b * b), 1e-12, 1.0)   # ||P_i x|| = cos theta_i
        s = np.sqrt(np.clip(1.0 - r * r, 0.0, None))      # 直交補空間成分 = sin theta_i
        theta = np.arctan2(s, r)
        return a, b, r, s, theta

    def contributions(self, X: np.ndarray, sigma: float) -> np.ndarray:
        """各大円の寄与 exp(-theta_i^2 / 2 sigma^2)  (M, N)。"""
        return np.exp(-self._angles(X)[4] ** 2 / (2.0 * sigma * sigma))

    def potential(self, X: np.ndarray, sigma: float) -> np.ndarray:
        return self.contributions(X, sigma).sum(1)

    def riemannian_grad(self, X: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """球面上の勾配 grad_S Phi(x) とポテンシャル値。

        dPhi/dx = sum_i w_i * theta_i / (sigma^2 sin theta_i) * P_i x / ||P_i x||
        (theta/sin theta -> 1 なので大円上でも有界)。最後に接空間へ射影する。
        """
        a, b, r, s, theta = self._angles(X)
        w = np.exp(-theta ** 2 / (2.0 * sigma * sigma))
        ratio = np.where(s > 1e-9, theta / np.maximum(s, 1e-12), 1.0)
        c = w * ratio / (sigma * sigma * r)
        G = (c * a) @ self.V + (c * b) @ self.U
        G -= np.sum(G * X, axis=1, keepdims=True) * X
        return G, w.sum(1)

    # -- seeds ---------------------------------------------------------------

    def pairwise_foci(self, max_pairs: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """全大円ペアの最近接点 (主角) の中点と、そのペア間角距離 delta_ij。

        平面 A_i=[v_i,u_i], A_j=[v_j,u_j] について C = A_i^T A_j (2x2) を SVD すると
        最大特異値 = cos(最小主角)、特異ベクトルが各大円上の最近接点を与える。
        """
        N = len(self.V)
        iu, ju = np.triu_indices(N, 1)
        if len(iu) > max_pairs:
            sel = rng.choice(len(iu), size=max_pairs, replace=False)
            iu, ju = iu[sel], ju[sel]
        VV, VU, UU = self.V @ self.V.T, self.V @ self.U.T, self.U @ self.U.T
        C = np.empty((len(iu), 2, 2))
        C[:, 0, 0] = VV[iu, ju]
        C[:, 0, 1] = VU[iu, ju]      # v_i · u_j
        C[:, 1, 0] = VU[ju, iu]      # u_i · v_j
        C[:, 1, 1] = UU[iu, ju]
        Lc, S, Rt = np.linalg.svd(C)
        p = Lc[:, 0, 0:1] * self.V[iu] + Lc[:, 1, 0:1] * self.U[iu]
        q = Rt[:, 0, 0:1] * self.V[ju] + Rt[:, 0, 1:2] * self.U[ju]
        foci = _normalize_rows(p + q)
        delta = np.arccos(np.clip(S[:, 0], -1.0, 1.0))
        return foci, delta

    def global_axis(self) -> np.ndarray:
        """sigma -> 大 の極限で Phi の極大となる方向 (= sum_i P_i の主固有ベクトル)。"""
        B = np.vstack([self.V, self.U])
        return np.linalg.svd(B, full_matrices=False)[2][0]


# =============================================================================
# 3. Concept-core extraction
# =============================================================================

@dataclass
class InterferenceConfig:
    sigma_scale: float = 0.5        # sigma = sigma_scale * median(大円ペア間の角距離)
    sigma: Optional[float] = None   # 絶対値で固定したい場合 (rad)
    max_pairs: int = 4096           # ペアワイズ交点のサンプリング上限
    n_seeds: int = 48               # 勾配上昇にかける候補点数 (Phi 上位から)
    ascent_steps: int = 20          # 局所勾配上昇のステップ数
    init_step: float = 0.5          # 初期ステップ長 (sigma 単位、適応的に増減)
    k_max: int = 3                  # 出力する概念核の最大数
    separation: float = 1.0         # 独立とみなす最小角距離 (sigma 単位)
    include_global_axis: bool = True
    seed: int = 0


@dataclass
class ConceptCore:
    vector: np.ndarray              # 概念核の刺激座標 (単位ベクトル)
    potential: float                # Phi(x*)
    resonance: float                # Phi(x*) / N  (全大円のうち何割が焦点に集束しているか)
    contributions: np.ndarray       # 各トークン大円の寄与 w_i (N,)

    def top_tokens(self, tokens: Sequence[str], k: int = 5) -> List[Tuple[str, float]]:
        order = np.argsort(-self.contributions)[:k]
        return [(tokens[i], float(self.contributions[i])) for i in order]


@dataclass
class ExtractionResult:
    text: str
    tokens: List[str]
    token_ids: List[int]
    cores: List[ConceptCore]
    sigma: float
    elapsed_ms: float = 0.0
    trace: Dict[str, float] = field(default_factory=dict)

    @property
    def primary(self) -> np.ndarray:
        return self.cores[0].vector

    def matrix(self) -> np.ndarray:
        return np.stack([c.vector for c in self.cores])


def extract_cores(V: np.ndarray, cfg: Optional[InterferenceConfig] = None
                  ) -> Tuple[List[ConceptCore], float, Dict[str, float]]:
    """単位ベクトル列 V (N, D) から干渉極大点 (多重概念核) を求める。"""
    cfg = cfg or InterferenceConfig()
    V = _normalize_rows(np.asarray(V, dtype=np.float64))
    N = len(V)
    if N == 1:
        return [ConceptCore(V[0].copy(), 1.0, 1.0, np.ones(1))], 0.0, {"n_seeds": 0}

    rng = np.random.default_rng(cfg.seed)
    fld = GreatCircleField(V, seed=cfg.seed)
    foci, delta = fld.pairwise_foci(cfg.max_pairs, rng)

    if cfg.sigma is not None:
        sigma = float(cfg.sigma)
    else:
        sigma = float(cfg.sigma_scale * np.median(delta))
    sigma = max(sigma, 1e-3)   # 全大円が一致する退化ケース (N=2 等) の保護

    seeds = foci
    if cfg.include_global_axis:
        seeds = np.vstack([seeds, fld.global_axis()[None, :]])
    phi0 = fld.potential(seeds, sigma)
    keep = np.argsort(-phi0)[:min(cfg.n_seeds, len(seeds))]
    X, phi = seeds[keep].copy(), phi0[keep].copy()

    # --- 局所リーマン勾配上昇 (適応ステップ + 単調増加保証) ---
    step = np.full(len(X), cfg.init_step * sigma)
    for _ in range(cfg.ascent_steps):
        G, _ = fld.riemannian_grad(X, sigma)
        gn = np.linalg.norm(G, axis=1, keepdims=True)
        Xn = _normalize_rows(X + step[:, None] * G / np.maximum(gn, 1e-12))
        phin = fld.potential(Xn, sigma)
        acc = phin > phi
        X[acc], phi[acc] = Xn[acc], phin[acc]
        step = np.where(acc, step * 1.2, step * 0.5)

    # --- 独立な極大点の選択 (対蹠点は同一視) ---
    min_sep = max(cfg.separation * sigma, 0.05)
    cos_sep = np.cos(min_sep)
    chosen: List[int] = []
    for i in np.argsort(-phi):
        if all(abs(X[i] @ X[j]) < cos_sep for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= cfg.k_max:
            break

    cores = []
    for i in chosen:
        x = X[i].copy()
        if np.sum(V @ x) < 0:      # Phi(x)=Phi(-x) なので入力トークン側へ向きを揃える
            x = -x
        w = fld.contributions(x[None, :], sigma)[0]
        cores.append(ConceptCore(x, float(phi[i]), float(phi[i] / N), w))
    trace = {"n_pairs": float(len(delta)), "median_delta": float(np.median(delta)),
             "n_seeds": float(len(keep))}
    return cores, sigma, trace


class ShallowConceptExtractor:
    """テキスト -> 多重概念核。教師モデルの Transformer 層は一切使わない。"""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        config: Optional[InterferenceConfig] = None,
        use_layer0_norm: bool = True,
        center: bool = True,
        trust_remote_code: bool = False,
        verbose: bool = True,
    ) -> None:
        from transformers import AutoTokenizer
        self.config = config or InterferenceConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.weights = ShallowWeights(model_id, use_layer0_norm=use_layer0_norm,
                                      center=center, verbose=verbose)

    def tokenize(self, text: str) -> Tuple[List[int], List[str]]:
        """入力文字列を構成する全トークン (バイトレベル BPE なので全文字を被覆)。
        ストップワード除去・頻度足切りは行わない。"""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        toks = [self.tokenizer.decode([i]) for i in ids]
        return ids, toks

    def token_vectors(self, text: str) -> Tuple[List[int], List[str], np.ndarray]:
        ids, toks = self.tokenize(text)
        return ids, toks, self.weights.unit_vectors(ids)

    def extract(self, text: str, config: Optional[InterferenceConfig] = None) -> ExtractionResult:
        t0 = time.perf_counter()
        ids, toks, V = self.token_vectors(text)
        if len(ids) == 0:
            raise ValueError("empty input")
        cores, sigma, trace = extract_cores(V, config or self.config)
        return ExtractionResult(text, toks, ids, cores, sigma,
                                (time.perf_counter() - t0) * 1e3, trace)

    def mean_vector(self, text: str) -> np.ndarray:
        """従来の浅層抽出 (全トークンの単純平均) — 比較用ベースライン。"""
        _, _, V = self.token_vectors(text)
        m = V.mean(0)
        return m / max(np.linalg.norm(m), 1e-12)


def cuda_was_initialized() -> bool:
    """この process で CUDA が初期化されたかどうか (torch 未 import なら False)。"""
    import sys
    torch = sys.modules.get("torch")
    return bool(torch is not None and torch.cuda.is_initialized())


if __name__ == "__main__":
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # スクリプト実行時は GPU を完全遮断
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    text = " ".join(sys.argv[1:]) or "We trained a neural network in Python to predict CRISPR off-target cuts in the genome."
    ex = ShallowConceptExtractor()
    res = ex.extract(text)
    print(f"\ntext   : {text}")
    print(f"tokens : {len(res.tokens)}  sigma={res.sigma:.4f} rad  time={res.elapsed_ms:.1f} ms")
    for k, c in enumerate(res.cores):
        tops = ", ".join(f"{t.strip() or repr(t)}:{w:.2f}" for t, w in c.top_tokens(res.tokens, 6))
        print(f"core#{k}: Phi={c.potential:.3f} resonance={c.resonance:.3f} | {tops}")
    print(f"CUDA initialized: {cuda_was_initialized()}")
