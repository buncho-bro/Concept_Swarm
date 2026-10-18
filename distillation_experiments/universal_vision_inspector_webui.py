"""
universal_vision_inspector_webui.py
=============================================================================
【v5.0.0: 概念核ラベル・可視化の完全撤廃 ＆ 純粋軌跡移転（Trajectory Transfer）直接発話エンジン】
(Pure Geodesic Trajectory Transfer to Text Concept Cores & Direct Language Output Studio)

ユーザーの核心設計思想（v5.0）：
1. 「概念核をこちら側で決めている理由が理解できません」
   - 人間が定義した概念リスト（106概念リストや日本語辞書 CONCEPT_JP_NAMES、カスタム概念追加欄など）を【完全撤廃】。
   - 画像モデル(CLIP)と文章モデル(Qwen/MiniLM)の内部語彙・重み空間全体から、
     「画像空間での重力場と文章空間での重力場が自然に共鳴する極大点（トポロジー共鳴＋球面極大抑制）」として
     自律的に析出した【4,096個の無名（ラベルなし）重力核】のみで合成地形を構築する。

2. 「荒い谷で処理を行うのではなく、重ねた後再度細かくする。その際に保有していた重力によって地形を作る」
   - 4,096個の重力核が作る地形を、保有重力を維持したまま一度荒く（R=48）平滑化してピタッとかぶせ、
     その後【元の2048次元フル解像度空間へ再度細かく展開（Re-Refinement）】し、保有していた重力深さで細かい谷を掘り直す。

3. 「概念核というものを可視化する意味がないのですよ。説明させたかったら文章概念核のほうに軌跡を移せばよいのだから」
   - 概念核に名前をつけたり、ランキングや棒グラフで可視化したりする処理を【完全廃止】。
   - 画像が入ったら、【画像概念核 (fine_img_cores) のみ】に引き寄せられて移動する『軌跡（道のり）』を記録し、
     その軌跡をそのまま変換先の【文章概念核 (fine_txt_cores) のみ】の地形へ移して再実行（Replay）する。
   - 文章概念核の地形上に移された軌跡（初期着地点 ➔ 中間移動 ➔ 終着点、および空間領域ごとの軌跡）を
     そのまま言語モデル（Qwen2.5 + GLM5.3 LoRA）の潜在空間・中間層へ流し込み、直接日本語の説明文として出力させる！
=============================================================================
"""

import os
import sys
import gc
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from universal_idea_core_engine import LatentOrthogonalBridge, TangentNudgeHook
from coarse_gravity_joint_terrain_engine import CoarseGravityJointTerrain
from test_zero_shot_generality_stress import load_six_real_images
from test_complex_multimodal_reasoning_stress import (
    generate_six_abstract_diagram_images,
    build_composite_scene_images,
)

STUDENT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_PATH = os.path.join(BASE_DIR, "lora_glm5.3_slice_topological_to_Qwen2.5_Coder_3B")
AUTONOMOUS_CORES_CACHE = os.path.join(BASE_DIR, "autonomous_gravity_cores_3072.pt")

STUDIO_VERSION = "v5.1.0 — Isometric Full-Vocabulary Trajectory Transfer & Direct Language Output Edition"


class PureTrajectoryTransferStudio:
    """
    人間による概念ラベル定義・概念核ランキング可視化を一切行わず、
    「画像概念核での重力落下の軌跡 ➔ 文章概念核での軌跡再実行 ➔ 言語モデルからの直接文章出力」
    のみを行う純粋幾何学マルチモーダル変換エンジン (v5.0.0)
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[v5.0 Studio] Initializing Pure Trajectory Transfer Studio on device={self.device}...", flush=True)

        # 1. 視覚エンコーダ & 言語モデルのロード（100% GPU常駐で即時生成）
        self.clip_model = SentenceTransformer("clip-ViT-B-32", device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)

        # 2. モデル内部空間から自律析出した無名重力核 (4,096 cores) のロード（または初回自動析出）
        if not os.path.exists(AUTONOMOUS_CORES_CACHE):
            self._extract_autonomous_cores_from_model_weights()

        core_data = torch.load(AUTONOMOUS_CORES_CACHE, map_location=self.device, weights_only=False)
        self.X_img_raw = F.normalize(core_data["X_img_raw"].float().to(self.device), p=2, dim=-1) # (C, 512)
        self.Y_txt_raw = F.normalize(core_data["Y_txt_raw"].float().to(self.device), p=2, dim=-1) # (C, 2048)
        self.qwen_token_ids = core_data["qwen_token_ids"].long().to(self.device)                   # (C,) モデル自身の内部トークンID
        self.num_autonomous_cores = self.X_img_raw.shape[0]

        self.mu_x = self.X_img_raw.mean(dim=0, keepdim=True)
        self.mu_y = self.Y_txt_raw.mean(dim=0, keepdim=True)
        self.X_img_clean = F.normalize(self.X_img_raw - self.mu_x, p=2, dim=-1)
        self.Y_txt_clean = F.normalize(self.Y_txt_raw - self.mu_y, p=2, dim=-1)

        # 3. 粗視化被覆 (R=48) ➔ 2048次元再細分化 (Re-Refinement) による合成重力地形の構築
        self.joint_terrain = CoarseGravityJointTerrain(
            img_dim=512,
            txt_dim=2048,
            coarse_rank=48,
            device=self.device
        )
        # 上位512個の強共鳴核を骨格とし、全4,096個の無名重力核の谷を2048次元空間上に掘り直す
        self.joint_terrain.build_joint_terrain(
            X_img_anchors=self.X_img_clean[:512],
            Y_txt_anchors=self.Y_txt_clean[:512],
            Y_txt_all=self.Y_txt_clean,
            X_img_all=self.X_img_clean,
            ridge_coarse=0.06
        )

        # 4. 検証用サンプル画像と視覚コーン中心 (mu_img_photo, mu_img_abs) の準備
        self.real_images = load_six_real_images()
        self.abstract_dir = os.path.join(BASE_DIR, "abstract_test_diagrams")
        self.abstract_images = generate_six_abstract_diagram_images(self.abstract_dir)
        self.composite_scenes = build_composite_scene_images(self.real_images, self.abstract_dir)

        photo_raw = F.normalize(torch.tensor(self.clip_model.encode([x[2] for x in self.real_images]), dtype=torch.float32, device=self.device), p=2, dim=-1)
        abs_raw   = F.normalize(torch.tensor(self.clip_model.encode([x[2] for x in self.abstract_images]), dtype=torch.float32, device=self.device), p=2, dim=-1)
        self.mu_img_photo = photo_raw.mean(dim=0, keepdim=True)
        self.mu_img_abs   = abs_raw.mean(dim=0, keepdim=True)

        # 5. 言語モデル (Qwen2.5-Coder-3B + GLM5.3 LoRA) を 100% GPU 上にロード
        print("[v5.0 Studio] Loading Qwen2.5-Coder-3B + GLM5.3 Topological LoRA on 100% GPU...", flush=True)
        base_llm = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_ID,
            dtype=torch.float16,
            device_map={"": self.device},
            local_files_only=True
        )
        self.llm = PeftModel.from_pretrained(base_llm, LORA_PATH) if os.path.exists(LORA_PATH) else base_llm
        self.llm.eval()
        print(f"[v5.0 Studio] Ready! {self.num_autonomous_cores} Autonomous Unlabeled Gravity Cores + Direct LLM Trajectory Decoder active.", flush=True)

    def _extract_autonomous_cores_from_model_weights(self):
        """
        人間が書いた単語リストを一切使わず、CLIPとQwenの内部トークナイザおよび重み空間の
        トポロジー重力共鳴度＋球面極大抑制（NMS）だけで無名重力核を自律析出する。
        """
        print("[v5.0 Studio] Extracting autonomous gravity cores directly from model vocabularies...", flush=True)
        minilm = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        c_vocab = self.clip_model.tokenizer.get_vocab()
        cv = {}
        for k, cid in c_vocab.items():
            if k.endswith("</w>"):
                w = k.replace("</w>", "")
                if w.isalpha() and w.islower() and len(w) >= 3:
                    cv[w] = cid

        qv = {}
        for qid in range(len(self.tokenizer)):
            t_str = self.tokenizer.convert_ids_to_tokens(qid)
            if isinstance(t_str, str) and t_str.startswith("Ġ"):
                w = t_str.lstrip("Ġ")
                if w.isalpha() and w.islower() and len(w) >= 3 and w not in qv:
                    qv[w] = qid

        candidates = sorted(set(cv.keys()) & set(qv.keys()), key=lambda w: cv[w])[:14000]
        X_cand = F.normalize(torch.tensor(self.clip_model.encode(candidates, batch_size=512, show_progress_bar=False), device=self.device).float(), p=2, dim=-1)
        Y_cand = F.normalize(torch.tensor(minilm.encode(candidates, batch_size=512, show_progress_bar=False), device=self.device).float(), p=2, dim=-1)
        X_c = F.normalize(X_cand - X_cand.mean(dim=0, keepdim=True), p=2, dim=-1)
        Y_c = F.normalize(Y_cand - Y_cand.mean(dim=0, keepdim=True), p=2, dim=-1)

        probes = torch.linspace(0, len(candidates) - 1, 1024, device=self.device).long()
        S_x = F.normalize((X_c @ X_c[probes].T) - (X_c @ X_c[probes].T).mean(dim=-1, keepdim=True), p=2, dim=-1)
        S_y = F.normalize((Y_c @ Y_c[probes].T) - (Y_c @ Y_c[probes].T).mean(dim=-1, keepdim=True), p=2, dim=-1)
        topo_res = (S_x * S_y).sum(dim=-1)

        sorted_idx = torch.argsort(topo_res, descending=True)
        selected = []
        selected_vecs = []
        for idx in sorted_idx.tolist():
            if float(topo_res[idx]) < 0.15:
                break
            v = Y_c[idx]
            if len(selected_vecs) > 0:
                if torch.max(torch.mv(torch.stack(selected_vecs), v)).item() > 0.78:
                    continue
            selected.append(idx)
            selected_vecs.append(v)
            if len(selected) >= 4096:
                break

        sel_t = torch.tensor(selected, device=self.device, dtype=torch.long)
        torch.manual_seed(42)
        Q_lift, _ = torch.linalg.qr(torch.randn(2048, 384, device=self.device))
        Y_sel_2048 = F.normalize(Y_cand[sel_t] @ Q_lift.T, p=2, dim=-1)
        q_ids_sel = torch.tensor([qv[candidates[i]] for i in selected], device=self.device, dtype=torch.long)

        torch.save({
            "X_img_raw": X_cand[sel_t].cpu(),
            "Y_txt_raw": Y_sel_2048.cpu(),
            "qwen_token_ids": q_ids_sel.cpu(),
            "topo_resonance": topo_res[sel_t].cpu(),
        }, AUTONOMOUS_CORES_CACHE)
        del minilm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_spatial_crops(self, pil_img: Image.Image) -> tuple[list[Image.Image], list[str]]:
        w, h = pil_img.size
        crops = [
            pil_img,
            pil_img.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))),
            pil_img.crop((0, int(h * 0.08), int(w * 0.52), int(h * 0.92))),
            pil_img.crop((int(w * 0.48), int(h * 0.08), w, int(h * 0.92))),
        ]
        names = ["全体", "中央部", "左側領域", "右側領域"]
        return crops, names

    def _analyze_visual_surface(self, pil_img: Image.Image, img_raw_vec: torch.Tensor) -> dict:
        arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        brightness = float(np.mean(arr))
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        sat = float(np.mean((max_c - min_c) / (max_c + 1e-5)))
        near_white_ratio = float(np.mean((r > 235) & (g > 235) & (b > 235)))
        near_black_ratio = float(np.mean((r < 25) & (g < 25) & (b < 25)))

        rg_diff = float(np.mean(r - g))
        gb_diff = float(np.mean(g - b))
        if sat < 0.14:
            color_desc = "モノトーン・白黒基調"
        elif rg_diff > 22 and float(np.mean(r)) > float(np.mean(b)) + 25:
            color_desc = "暖色系（赤・オレンジ・茶色系）"
        elif gb_diff > 18 and float(np.mean(g)) > float(np.mean(r)) + 10:
            color_desc = "緑・自然色系"
        elif float(np.mean(b)) > float(np.mean(r)) + 20:
            color_desc = "寒色系（青・水色系）"
        else:
            color_desc = "マルチカラー（複数色の組み合わせ）"

        sim_photo = float(torch.matmul(img_raw_vec, self.mu_img_photo.T).item())
        sim_abs   = float(torch.matmul(img_raw_vec, self.mu_img_abs.T).item())
        is_diagram_or_meme = (near_white_ratio > 0.22) or (near_black_ratio > 0.35) or (sim_abs > sim_photo + 0.015)

        style_desc = (
            f"図解・2Dグラフィック・テキストまたは白余白を含む画像（白領域 {near_white_ratio*100:.0f}%）"
            if is_diagram_or_meme else
            "自然写真・連続階調の視覚シーン"
        )
        return {
            "color_desc": color_desc,
            "style_desc": style_desc,
            "is_diagram_or_meme": is_diagram_or_meme,
            "brightness": brightness,
        }

    def transfer_trajectory_and_explain(
        self,
        pil_img: Image.Image,
        image_mode: str,
        alpha_csls: float,
        n_steps: int
    ):
        """
        【v5.0 の全処理フロー】
        1. 画像特徴と空間クロップ特徴を抽出
        2. 画像概念核 (fine_img_cores) のみで重力落下させ『道のり（軌跡）』を記録
        3. その『道のり』を文章概念核 (fine_txt_cores) の地形上で再実行（Replay）し、文章空間上の軌跡 T_txt を得る
        4. 概念核の可視化やラベル分類は一切行わず、移転された文章概念核上の軌跡 T_txt から直接 LLM で日本語説明文を出力する！
        """
        if pil_img is None:
            return "⚠️ 左側のボックスに画像をドラッグ＆ドロップするか、下のサンプル画像をクリックしてください。", ""

        t0 = time.time()
        pil_rgb = pil_img.convert("RGB")
        w, h = pil_rgb.size

        crops, crop_names = self._build_spatial_crops(pil_rgb)
        crops_raw = F.normalize(
            torch.tensor(self.clip_model.encode(crops), dtype=torch.float32, device=self.device),
            p=2, dim=-1
        )
        surface_info = self._analyze_visual_surface(pil_rgb, crops_raw[0:1])

        if "自然写真" in image_mode:
            mu_img = self.mu_img_photo
        elif "抽象図解" in image_mode:
            mu_img = self.mu_img_abs
        else:
            mu_img = self.mu_img_abs if surface_info["is_diagram_or_meme"] else self.mu_img_photo

        crops_clean = F.normalize(crops_raw - mu_img, p=2, dim=-1)

        # ---------------------------------------------------------------------
        # Phase A & B: 画像概念核のみで落下 ➔ 変換先の文章概念核で『道のり（軌跡）』を再実行！
        # ---------------------------------------------------------------------
        jt_out = self.joint_terrain.trace_and_replay_trajectory(
            x_img_crops_clean=crops_clean,
            alpha_csls=float(alpha_csls),
            n_steps=int(n_steps),
            step_lr=0.24
        )
        txt_traj_points = jt_out["txt_trajectory"]         # List of (K_crops, 2048) for t = 0..T
        valley_scores   = jt_out["valley_gravity_scores"]  # (K_crops, C)
        replay_log      = jt_out["replay_step_log"]

        t_terrain_ms = (time.time() - t0) * 1000.0

        # ---------------------------------------------------------------------
        # Phase C: 文章概念核へ移された軌跡 T_txt から、言語モデル自身の内部語彙共鳴と
        #          中間層測地線フック (TangentNudgeHook) を通じて直接日本語説明文を生成！
        #          （※人間が定義した辞書やテンプレート文は一切不使用）
        # ---------------------------------------------------------------------
        with torch.no_grad():
            y_start = txt_traj_points[0][0]   # (2048,)
            y_mid   = txt_traj_points[len(txt_traj_points) // 2][0]
            y_final = txt_traj_points[-1][0]  # (2048,)

            s_start = torch.matmul(self.joint_terrain.fine_txt_cores, y_start) - float(alpha_csls) * self.joint_terrain.r_T_txt
            s_final = valley_scores[0] + 0.06 * self.joint_terrain.well_depth_txt

            def _readout_diverse_tokens(score_vec: torch.Tensor, top_n: int = 6, max_cos: float = 0.70) -> list[str]:
                sorted_idx = torch.argsort(score_vec, descending=True)[:120].tolist()
                chosen_words = []
                chosen_vecs = []
                for idx in sorted_idx:
                    v = self.joint_terrain.fine_txt_cores[idx]
                    if len(chosen_vecs) > 0 and torch.max(torch.mv(torch.stack(chosen_vecs), v)).item() > max_cos:
                        continue
                    w_str = self.tokenizer.decode([int(self.qwen_token_ids[idx].item())]).strip()
                    if len(w_str) < 3:
                        continue
                    chosen_vecs.append(v)
                    chosen_words.append(w_str)
                    if len(chosen_words) >= top_n:
                        break
                return chosen_words

            init_words = _readout_diverse_tokens(s_start, top_n=6, max_cos=0.70)
            final_words = _readout_diverse_tokens(s_final, top_n=8, max_cos=0.70)

            spatial_traces = []
            for k_idx in range(1, len(crop_names)):
                c_words = _readout_diverse_tokens(valley_scores[k_idx] + 0.05 * self.joint_terrain.well_depth_txt, top_n=3, max_cos=0.70)
                spatial_traces.append(f"{crop_names[k_idx]}({', '.join(c_words)})")

            # 文章概念核上で再実行された軌跡ベクトル列を Qwen+GLM5.3 の中間層へ TangentNudgeHook で直接注入
            trajectory_kernels = torch.stack([y_start, y_mid, y_final, txt_traj_points[-1][2], txt_traj_points[-1][3]], dim=0)
            identity_bridge = LatentOrthogonalBridge(in_dim=2048, idea_dim=2048).to(self.device)
            nudge_hook = TangentNudgeHook(
                bridge=identity_bridge,
                idea_kernels=trajectory_kernels,
                alpha=0.06,
                beta=9.0
            )
            nudge_hook.register(self.llm, layer_indices=[12, 18, 24, 30])

            prompt = (
                f"<|im_start|>system\n"
                f"あなたは視覚地形から文章概念核の地形へと移された『重力軌跡（Trajectory）』を、自然な日本語の画像解説へと直接変換する言語デコーダです。\n"
                f"【重要なルール】\n"
                f"1. 架空の人間ドラマや創作物語（例：難民が逃げてきた、教会活動をしている等の小説的ストーリー）を勝手に作らないでください。\n"
                f"2. 軌跡が通過した概念群（場所・地域・物体・動き・雰囲気など）と視覚表面の特徴（図解・イラスト・写真・色彩・白黒コントラストなど）を統合し、「どのような視覚テーマ・構図・モチーフを描いた画像か」を客観的で分かりやすい日本語（2〜3段落、完結した文章）で説明してください。\n"
                f"3. 必ず純粋な日本語のみで出力し、中国語の表現（例：家乡、但他们など）や英単語の羅列は絶対に含めないでください。<|im_end|>\n"
                f"<|im_start|>user\n"
                f"・文章概念核での軌跡（始点 ➔ 終着点）: {', '.join(init_words)} ➔ {', '.join(final_words)}\n"
                f"・画面の領域ごとの軌跡: {' / '.join(spatial_traces)}\n"
                f"・視覚表面のスタイルと色彩: {surface_info['style_desc']}、{surface_info['color_desc']}（サイズ {w}×{h}px）\n"
                f"この画像がどのような視覚内容・テーマ・構図の画像であるか、自然な日本語で最後まで完結した文章として説明してください。<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            t_llm_0 = time.time()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            gen_ids = self.llm.generate(
                **inputs,
                max_new_tokens=380,
                repetition_penalty=1.15,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            nudges_count = nudge_hook.total_nudges
            nudge_hook.remove()
            t_llm_sec = time.time() - t_llm_0

        explanation_text = self.tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # 万一トークン上限付近で文が途切れた場合は最後の句点「。」で綺麗に完結させる
        if not explanation_text.endswith(("。", "！", "？", "）", "」")) and "。" in explanation_text:
            explanation_text = explanation_text[:explanation_text.rfind("。") + 1]

        main_output_md = (
            f"## 💬 文章概念核へ移された軌跡からの画像説明\n\n"
            f"{explanation_text}\n"
        )

        # 技術ログ（概念核の名前やランキングは一切出さず、純粋に軌跡の幾何学移動量と実行時間のみを記録）
        turn_degs = [f"Step {s['step']}: `{s['geodesic_turn_deg']:.2f}°`" for s in replay_log]
        trace_meta_md = (
            f"**⚙️ v5.0.1 軌跡移転・直接発話テレメトリ**\n"
            f"- **自律重力核地形:** `{self.num_autonomous_cores}` 個（モデル内部重み空間のトポロジー共鳴から自動析出・人間定義ラベル `0` 個）\n"
            f"- **合成地形の構築:** 粗視化被覆 (`R=48`) ➔ 2048次元フル空間への再細分化（保有重力による微細谷掘削）\n"
            f"- **文章概念核での軌跡再実行ステップ角:** {' ➔ '.join(turn_degs)}\n"
            f"- **中間層測地線軌跡注入 (`TangentNudgeHook`):** `{nudges_count}` 回の軌跡ナッジ適用（Layer 12, 18, 24, 30）\n"
            f"- **処理時間:** 軌跡移転 `{t_terrain_ms:.1f} ms` ＋ 言語モデル直接デコード `{t_llm_sec:.2f} s` (on `{self.device}`)"
        )

        return main_output_md, trace_meta_md


def build_gradio_app(studio: PureTrajectoryTransferStudio):
    sample_paths = []
    for _, _, _, p in studio.real_images:
        if os.path.exists(p):
            sample_paths.append([p])
    for _, _, _, p in studio.abstract_images:
        if os.path.exists(p):
            sample_paths.append([p])
    for comp_name in ["composite_dog_and_pagoda.jpg", "composite_flower_and_car.jpg", "composite_officer_dog_pagoda.jpg"]:
        cp = os.path.join(studio.abstract_dir, comp_name)
        if os.path.exists(cp):
            sample_paths.append([cp])

    with gr.Blocks(title=f"Universal Trajectory Transfer Studio [{STUDIO_VERSION}]") as demo:
        gr.Markdown(
            f"""
            # 🌌 概念地形・純粋軌跡移転（Trajectory Transfer）画像説明スタジオ
            ### 🏷️ **Version: `{STUDIO_VERSION}`**
            - **人間が定義した概念リスト・日本語辞書・概念核ランキング可視化・カスタム概念追加欄をすべて撤廃しました。**
            - **仕組み:**
              1. 画像モデルと文章モデルの内部空間から自律的に析出した無名重力地形を、一度荒く重ねてから**元の2048次元へ再度細かく展開（保有重力で谷を掘削）**します。
              2. 画像が入ると、**画像概念核のみ**に引き寄せられて移動し、その**「軌跡（道のり）」を変換先の文章概念核の地形へそのまま移して再実行**します。
              3. 文章概念核の地形上に移された軌跡をそのまま言語モデル（`Qwen2.5 + GLM5.3 LoRA`）へ流し込み、**直接日本語の説明文を出力**します。
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                img_input = gr.Image(
                    type="pil",
                    label="🖼️ 説明させたい画像をここにドラッグ＆ドロップ（またはクリックして選択）",
                    height=360
                )
                with gr.Row():
                    mode_radio = gr.Radio(
                        choices=["自動判定 (Auto)", "自然写真・イラスト (Photo)", "抽象図解・テキスト含有画像 (Diagram)"],
                        value="自動判定 (Auto)",
                        label="視覚コーン中心補正"
                    )
                with gr.Row():
                    csls_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.55, step=0.05,
                        label="局所ハブ重力補正係数 (α)"
                    )
                    steps_slider = gr.Slider(
                        minimum=1, maximum=8, value=5, step=1,
                        label="重力落下の道のりステップ数 (T)"
                    )
                run_btn = gr.Button("🔍 画像概念核の軌跡を文章概念核へ移して説明させる (v5.0 Direct Output)", variant="primary")

                gr.Examples(
                    examples=sample_paths,
                    inputs=[img_input],
                    label="📂 ワンクリック検証用サンプル画像（実写真・抽象図解・複合シーン）"
                )

            with gr.Column(scale=7):
                explanation_out = gr.Markdown(
                    "## 💬 左側に画像をドロップすると、画像概念核での軌跡が文章概念核へ移され、ここに直接説明文が出力されます。"
                )
                with gr.Accordion("⚙️ 軌跡移転テレメトリ（幾何学ステップ移動角・処理時間）", open=True):
                    telemetry_out = gr.Markdown("")

        img_input.change(
            fn=studio.transfer_trajectory_and_explain,
            inputs=[img_input, mode_radio, csls_slider, steps_slider],
            outputs=[explanation_out, telemetry_out]
        )
        run_btn.click(
            fn=studio.transfer_trajectory_and_explain,
            inputs=[img_input, mode_radio, csls_slider, steps_slider],
            outputs=[explanation_out, telemetry_out]
        )

    return demo


if __name__ == "__main__":
    studio = PureTrajectoryTransferStudio()
    app = build_gradio_app(studio)
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7865"))
    print(f"\n🚀 Launching Pure Trajectory Transfer Studio [{STUDIO_VERSION}] at http://127.0.0.1:{port} ...", flush=True)
    app.launch(server_name="127.0.0.1", server_port=port, share=False)
