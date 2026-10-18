# 粗視化被覆・再細分化重力地形 ＆ 純粋軌跡移転（Trajectory Transfer）仕様書 (v5.1)
## 人間定義ラベルの完全撤廃と「同モダリティ重力落下 ➔ 異モダリティ軌跡再実行」による直接発話アーキテクチャ

---

## 1. 根本思想とパラダイム転換（人間による概念核指定の排除）

### 1.1 従来のマルチモーダル統合における3つの罠
これまでの視覚（Vision）と言語（Language）の概念空間統合実験（v1.0〜v4.0）および一般的なマルチモーダルAIにおいて、以下の根本的な問題が存在していた：

1. **人間側による「概念核リスト・属性リスト」の決め打ちの危険性:**
   - 旧来の蒸留（25英単語アンカー）やゼロショット分類実験（人間が選んだ106概念リスト、あるいは色・形・材質などの64原始属性リスト）のように、**人間側が「どのような概念核を置くか」を指定した瞬間、モデルの認知空間は人間が用意した枠内に閉じ込められる**。
   - リストにない未知の事象（例：スナック菓子の箱やパロディ画像）が入力されると、既存の106個の谷のどれか（例：電子回路基板など）へ無理やり押し込める空間的歪みが発生する。
2. **「概念核を可視化・ラベル付けする」こと自体の無意味さ:**
   - 概念核（Concept Core）とは本来、高次元多様体内部における**重力の支点（局所高密度アトラクター座標）**であり、人間向けのラベル（「猫」「車」「回路」など）を貼ってランキング棒グラフで可視化するためのものではない。
   - **「画像がどんなものか説明させたければ、概念核を可視化するのではなく、画像概念核で描かれた『軌跡（道のり）』を文章概念核のほうに移せばよい」**（ユーザー提言）。
3. **「荒い谷」のままでの処理による微細構造の喪失:**
   - 異なるモデル空間（512次元の画像空間と2048次元の文章空間）は、荒く（低次元平滑化）すれば大局的な骨格表面をピタリと重ね合わせることができる。しかし、**荒い谷のまま推論を行うと個別の微細な意味構造が潰れてしまう**。

---

### 1.2 コア設計原則（v5.1 アーキテクチャ）

> **設計原則（ユーザー提言）：**
> 1. 「原始的な概念核をこちら側で指定するのは危険です。概念核をこちら側で決めている理由が理解できません。」
> 2. 「荒い谷で処理を行うのではなく、重ねた後**再度細かくする**。その際に保有していた重力によって地形を作る。」
> 3. 「画像が入れば**画像概念核のみ**、文章なら**文章概念核のみ**に引き寄せられ、その**道のり**を変換先の概念核で再度実行し、それを出力する。」
> 4. 「概念核というものを可視化する意味がないのですよ。説明させたかったら文章概念核のほうに軌跡を移せばよいのだから。」

---

## 2. システムアーキテクチャ全体像

```mermaid
flowchart TD
    subgraph Step0 [0. 自律重力核の析出 (人間定義ラベル 0個)]
        V_Model[画像モデル内部空間<br/>CLIP 512d] --> Coherence[トポロジー重力共鳴フィルタ<br/>11,845個の無名固有重力核]
        T_Model[言語モデル内部空間<br/>Qwen / MiniLM 2048d] --> Coherence
    end

    subgraph Step1_3 [1. 粗視化被覆 ➔ 2048次元再細分化 (CoarseGravityJointTerrain)]
        Coherence --> GravMeasure[Step 1: 保有重力質量 m_i と<br/>微細残差深さ g_i の計測・保持]
        GravMeasure --> CoarseOverlap[Step 2: 熱拡散平滑化と R=48 粗視化表面での<br/>直交プロクラステス完全被覆 R_coarse]
        CoarseOverlap --> ReRefine[Step 3: 再度細かくする (Re-Refinement)<br/>半直交等長写像 P_iso と保有重力 g_i により<br/>2048次元フル空間上に微細な谷を掘削]
        ReRefine --> Coexist[2048次元合成地形上に共存する2つの無名概念核群<br/>・画像概念核 fine_img_cores (11,845 × 2048)<br/>・文章概念核 fine_txt_cores (11,845 × 2048)]
    end

    subgraph PhaseA_B [2. 同モダリティ重力落下 ➔ 異モダリティ『道のり』再実行]
        ImgIn[入力画像 + 空間領域クロップ] --> InitMap[2048次元合成地形への初期着地 z_img^0]
        Coexist --> PhaseA[Phase A: 画像概念核 fine_img_cores のみに<br/>引き寄せられる測地線落下 t=1..T<br/>『道のり（重力比系列 w^1..w^T）』を記録]
        InitMap --> PhaseA
        PhaseA --> PhaseB[Phase B: 記録した『道のり w^1..w^T』を<br/>変換先の文章概念核 fine_txt_cores のみで再実行!<br/>文章空間上の軌跡 T_txt = y_txt^0 -> ... -> y_txt^T]
    end

    subgraph PhaseC [3. 概念核を可視化せず、移された軌跡から直接発話]
        PhaseB --> Hook[TangentNudgeHook による中間層測地線誘導<br/>Layer 12, 18, 24, 30 への軌跡ベクトル直接注入]
        PhaseB --> Readout[球面多様性読み出し (Spherical NMS)<br/>軌跡 T_txt が通過した言語モデル内部トークン共鳴]
        Hook --> LLM[Qwen2.5-Coder-3B + GLM-5.3 Slice Topological LoRA]
        Readout --> LLM
        LLM --> Output[自然な日本語による画像説明文を直接出力]
    end
```

---

## 3. 数学的定式化

### 3.1 保有重力の計測と粗視化被覆（Step 1 & Step 2）
画像空間の無名核集合を $\mathbf{X} \in \mathbb{R}^{C \times 512}$、文章空間の無名核集合を $\mathbf{Y} \in \mathbb{R}^{C \times 2048}$（$C = 11,845$）とする。
各空間の最大直径 $k = \max_{i,j} \|\mathbf{x}_i - \mathbf{x}_j\|_2$ で正規化した無次元相対距離行列 $\mathbf{M} = \mathbf{D} / k$ から、ガウス重力カーネル $\mathbf{K}_{\text{grav}} = \exp(-\mathbf{M}^2 / 2\tau^2)$ と各核の**保有重力質量** $m_i = \frac{1}{C}\sum_{j \neq i} K_{\text{grav}, ij}$ を算出する。

熱拡散オペレータ $\mathbf{P}_{\text{diff}} = \mathbf{D}^{-1}\mathbf{K}_{\text{grav}}$ によって高周波の凹凸を一度平滑化し、上位 $R=48$ 個の主軸基底 $\mathbf{V}_{\text{img, coarse}} \in \mathbb{R}^{512 \times R}$, $\mathbf{V}_{\text{txt, coarse}} \in \mathbb{R}^{2048 \times R}$ 上で重力質量重み付き直交プロクラステス変換 $\mathbf{R}_{\text{coarse}} \in SO(R)$ を解くことで、2つの荒い表面をピタリと重ね合わせる。

### 3.2 半直交等長写像による「再度細かくする（Re-Refinement）」（Step 3）
荒い表面のまま処理するのではなく、重ね合わせた後に元の **2048次元フル解像度空間** へ再度細かく展開する。
このとき、最小二乗回帰による細部距離の歪みを防ぐため、全核の重力重み付き相互共分散行列の特異値分解（SVD）から**半直交等長写像（Semi-Orthogonal Isometric Embedding）** $\mathbf{P}_{\text{iso}} \in \mathbb{R}^{512 \times 2048}$ を導出する：

$$\mathbf{U}_{\text{fine}} \boldsymbol{\Sigma} \mathbf{V}_{\text{fine}}^\top = (\mathbf{X} \odot \sqrt{\mathbf{m}})^\top (\mathbf{Y} \odot \sqrt{\mathbf{m}}), \qquad \mathbf{P}_{\text{iso}} = \mathbf{U}_{\text{fine}} \mathbf{V}_{\text{fine}}^\top \quad (\mathbf{P}_{\text{iso}} \mathbf{P}_{\text{iso}}^\top = \mathbf{I}_{512})$$

$\mathbf{P}_{\text{iso}} \mathbf{P}_{\text{iso}}^\top = \mathbf{I}_{512}$ が厳密に成り立つため、512次元空間内の任意の2点間の内積・距離は2048次元空間上でも **100% 無歪みで保存** される。
荒い共通表面 $\mathbf{S}_{2048}$ に対し、粗視化時に引っかかっていた**保有重力深さ** $g_i = m_i \cdot \rho_i \sqrt{1 - \|\mathbf{u}_{\text{coarse}, i}\|^2}$ と微細構造ベクトル $\mathbf{d}_{\text{fine}, i}$ を解放して掘り込むことで、2048次元超球面上に細かい谷を持つ**画像概念核 $\mathbf{C}_{\text{img}} \in \mathbb{R}^{C \times 2048}$** と **文章概念核 $\mathbf{C}_{\text{txt}} \in \mathbb{R}^{C \times 2048}$** が構築される。

### 3.3 同モダリティ重力落下と、変換先概念核での『道のり』の再実行（Phase A & Phase B）
1. **Phase A（画像概念核のみでの重力落下）:**
   入力画像ベクトル $\mathbf{z}_{\text{img}}^{(0)} \in \mathbb{S}^{2047}$ は、文章概念核には一切触れず、**画像概念核 $\mathbf{C}_{\text{img}}$ のみ**が作る重力場の中でステップ $t = 1 \dots T$ と測地線に沿って落下し、各ステップでの重力引き寄せ比率（道のり） $\mathbf{w}^{(t)} \in \Delta^{C-1}$ を記録する：
   $$\mathbf{w}^{(t)} = \text{softmax}\!\Big(\beta_t \big(\mathbf{z}_{\text{img}}^{(t-1)} \mathbf{C}_{\text{img}}^\top - \alpha_{\text{csls}} \mathbf{r}_{T, \text{img}} + \gamma \mathbf{g}_{\text{img}}\big)\Big)$$
   $$\mathbf{a}_{\text{img}}^{(t)} = \text{Normalize}\!\big(\mathbf{w}^{(t)} \mathbf{C}_{\text{img}}\big), \qquad \mathbf{z}_{\text{img}}^{(t)} = \text{Normalize}\!\Big(\mathbf{z}_{\text{img}}^{(t-1)} + \eta \big(\mathbf{a}_{\text{img}}^{(t)} - \langle \mathbf{a}_{\text{img}}^{(t)}, \mathbf{z}_{\text{img}}^{(t-1)} \rangle \mathbf{z}_{\text{img}}^{(t-1)}\big)\Big)$$

2. **Phase B（変換先の文章概念核での『道のり』の再実行）:**
   Phase A で記録された『道のり $\{\mathbf{w}^{(1)}, \dots, \mathbf{w}^{(T)}\}$』を、今度は変換先である **文章概念核 $\mathbf{C}_{\text{txt}}$ のみ**の地形上でステップ $t = 1 \dots T$ とそのまま再実行（Replay）する：
   $$\mathbf{a}_{\text{txt}}^{(t)} = \text{Normalize}\!\big(\mathbf{w}^{(t)} \mathbf{C}_{\text{txt}}\big), \qquad \mathbf{y}_{\text{txt}}^{(t)} = \text{Normalize}\!\Big(\mathbf{y}_{\text{txt}}^{(t-1)} + \eta \big(\mathbf{a}_{\text{txt}}^{(t)} - \langle \mathbf{a}_{\text{txt}}^{(t)}, \mathbf{y}_{\text{txt}}^{(t-1)} \rangle \mathbf{y}_{\text{txt}}^{(t-1)}\big)\Big)$$
   こうして得られた文章概念核上の連続軌跡 $\mathcal{T}_{\text{txt}} = (\mathbf{y}_{\text{txt}}^{(0)}, \mathbf{y}_{\text{txt}}^{(1)}, \dots, \mathbf{y}_{\text{txt}}^{(T)})$ を、`TangentNudgeHook` と球面多様性読み出しを通じて言語モデル（`Qwen2.5-Coder-3B + GLM-5.3 Slice Topological LoRA`）へ直接流し込み、日本語の説明文を出力する。

---

## 4. 実験的検証と重要な知見

| バージョン / フェーズ | 構成とアプローチ | 実証結果・発見された知見 |
| :--- | :--- | :--- |
| **GLM-5.3 0VRAM 全語彙大円干渉 & スライス蒸留** (`slice_topological_distiller.py`) | 辞書リストを全廃し、GLM-5.3 の全 154,880 語 × 6144d から 0MB VRAM で 54 個の自律核を析出。主軸等分位スライスと空間直径 $k$ による無次元化 $M = D/k$ で Qwen-3B へ同相蒸留。 | テキスト入力 0 語でトポロジー誤差を **97.86% 削減**（`0.32495` ➔ `0.00694`）。 |
| **v1.0〜v3.0: 106概念谷ベンチマークと複合シーン分解** (`test_complex_multimodal_reasoning_stress.py`) | Subspace-CSLS + 空間マルチクロップ直交残差分解 (OMP) + `TangentNudgeHook` (100% GPU `cuda:0`)。 | 複合シーン分解 Recall **91.7% (11/12)**、抽象図解 Top-3 **100% (6/6)**、ブラインド画像➔Pythonコード合成 `exec()` 合格率 **80.0%**（旧Rank-8は 0.0%）。 |
| **v4.0: 粗視化被覆・再細分化と『道のり』再実行** (`coarse_gravity_joint_terrain_engine.py`) | $R=28$ 粗視化被覆後に 2048次元へ再度細かく展開し、同モダリティ落下軌跡を異モダリティ核で再実行。 | 2048次元合成地形上での画像核・文章核の谷一致度 **`0.9051`** を達成。ただし人間定義の106概念リストが残存していた。 |
| **v5.0〜v5.1: 人間定義ラベル・可視化の全廃と 11,845 固有核フル空間軌跡移転** (`universal_vision_inspector_webui.py`) | 人間定義リスト・辞書・ランキング可視化を全廃。モデル内部の 11,845 固有語彙核を $\mathbf{P}_{\text{iso}}$ で無歪み再細分化し、移転された軌跡から直接 LLM が日本語説明を生成。 | ・実写（犬や寺院など）では人間定義ラベルなしで「自然の景色の中で犬が遊んでいる」等と直接発話。<br/>・日本語の地理矛盾ミーム画像（千葉県の緑のシルエット＋茨城県・群馬県・SAITAMA＋集中線）では、軌跡が自律的に `hometown` / `town`（地域・地元）や `election` / `voted`（区分マップ）、`mission` / `fight`（集中線スローガン構図）を通過し、「地域のつながりや対立・スローガン文字の配置」として出力。同時に、英語圏視覚モデル（`CLIP-ViT-B-32`）が「文字の存在と勢いは認識できるが、日本語漢字の意味や県境の固有知識までは持っていない」という視覚AI側の知識境界も幾何学的に可視化された。 |
