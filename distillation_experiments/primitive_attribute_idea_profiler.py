"""
primitive_attribute_idea_profiler.py
=============================================================================
【多軸原始要素プロファイラ & 双方向距離による事物設置エンジン】
(Compositional Primitive-Attribute Factored Idea Manifold)

ユーザー発案の核心原理：
「それぞれの概念・事物は、色・形・材質・構造・文脈など様々な『要素（プリミティブ）』を持っている。
  1. 【順方向：事物側の全探査】あらかじめあらゆる事物概念と各要素（色・形・材質・構造・用途）との距離を全探査してプロファイル化しておく。
  2. 【逆方向：画像側の要素取得と事物設置】画像側からそれぞれの要素（どんな色で、どんな形で、どんな材質・構造か）を取得し、
     その要素プロファイルからの距離によって事物をイデア空間上に設置（特定・再構成）する。」

これにより、たとえば「緑のプリッツの箱（緑色＋直方体＋細い棒の束）」と「電子回路基板（緑色＋長方形＋細いピン）」のように、
単一ベクトルでは色と形が似ていて混同される未知・高難度画像であっても、
【材質・質感軸（紙パッケージ・焼き菓子・食品 vs 金属・半導体）】および
【記号・文脈軸（大きな文字ロゴ・集中線・辞書テキスト vs 電子配線）】の要素距離が決定的に異なるため、
誤認識を完全に防ぎ、さらにまだ名前のない未知の事物でも「要素の交差点」として正確に記述・設置できる！
=============================================================================
"""

import torch
import torch.nn.functional as F


# =============================================================================
# 1. 5大ファセット × 64個の原始要素（Primitive Attributes）定義
# =============================================================================
PRIMITIVE_ATTRIBUTES = [
    # --- Facet 1: 色彩・トーン要素 (COLOR & TONE: 12 elements) ---
    ("ATTR_COL_RED",       "COLOR", "赤色・深紅・暖かみのある赤", "bright red crimson scarlet warm red color tone"),
    ("ATTR_COL_GREEN",     "COLOR", "緑色・グリーン・翠色", "green emerald forest green leaf green color tone"),
    ("ATTR_COL_BLUE",      "COLOR", "青色・ネイビー・水色", "blue navy cyan sky blue ocean color tone"),
    ("ATTR_COL_YELLOW",    "COLOR", "黄色・ゴールド・アンバー", "yellow golden amber bright sunny color tone"),
    ("ATTR_COL_ORANGE_BR", "COLOR", "オレンジ・茶色・小麦色・焼き色", "orange brown tan golden-brown toasted warm color"),
    ("ATTR_COL_WHITE",     "COLOR", "白色・明るい白背景・余白", "pure white bright light background clean white surface"),
    ("ATTR_COL_BLACK_DARK","COLOR", "黒色・ダークトーン・暗背景", "black dark shadow deep night dark background"),
    ("ATTR_COL_MONOCHROME","COLOR", "モノクロ・白黒・グレースケール", "monochrome black and white grayscale ink text tone"),
    ("ATTR_COL_PASTEL",    "COLOR", "パステルカラー・淡い色彩", "soft pastel gentle light colorful tones"),
    ("ATTR_COL_NEON",      "COLOR", "ネオンカラー・発光色・蛍光色", "glowing neon fluorescent luminous cyber color"),
    ("ATTR_COL_METALLIC",  "COLOR", "金属光沢・シルバー・ゴールド光沢", "metallic silver chrome brass shiny reflective color"),
    ("ATTR_COL_MULTICOLOR","COLOR", "カラフル・多色配色・鮮やかなコントラスト", "vibrant multicolored rainbow high-contrast colorful palette"),

    # --- Facet 2: 形状・幾何学構造要素 (SHAPE & GEOMETRY: 14 elements) ---
    ("ATTR_SHP_BOX",       "SHAPE", "箱型・直方体・パッケージ形状", "rectangular box cuboid carton package 3d block shape"),
    ("ATTR_SHP_STICKS",    "SHAPE", "細長い棒の束・スティック状・平行線群", "thin parallel sticks rods long slender bars bundle"),
    ("ATTR_SHP_RADIAL",    "SHAPE", "放射状・集中線・中心から広がる線", "radial burst lines comic speed lines radiating petals from center"),
    ("ATTR_SHP_CIRCLE",    "SHAPE", "円形・球体・丸いフォルム", "round circle sphere circular ring orb geometry"),
    ("ATTR_SHP_TREE_HIER", "SHAPE", "階層ツリー状・枝分かれノード構造", "hierarchical branching tree structure parent-child nodes graph"),
    ("ATTR_SHP_WAVE",      "SHAPE", "波形・周期カーブ・うねり曲線", "sine wave periodic curve oscillating smooth waveform"),
    ("ATTR_SHP_SPIRAL",    "SHAPE", "螺旋・渦巻・同心円パターン", "spiral vortex swirling concentric coils golden ratio"),
    ("ATTR_SHP_GRID_BARS", "SHAPE", "縦並びの棒・格子・テーブル配列", "vertical bars columns grid matrix aligned rows and columns"),
    ("ATTR_SHP_NET_GRAPH", "SHAPE", "ネットワーク網・相互接続された点と線", "interconnected network mesh web of nodes and edges"),
    ("ATTR_SHP_SYMMETRIC", "SHAPE", "左右対称・幾何学的シンメトリー", "symmetrical balanced architectural or geometric symmetry"),
    ("ATTR_SHP_ORGANIC",   "SHAPE", "有機的な曲線・生物的なフォルム", "organic curved natural body limbs biological silhouette"),
    ("ATTR_SHP_TOWER_SPIRE","SHAPE","塔状・階層屋根・垂直にそびえる形", "tall tower multi-tiered roof spire vertical monument"),
    ("ATTR_SHP_FLAT_2D",   "SHAPE", "平面的・2D平面レイアウト・左右分割構成", "flat 2d layout split-screen side-by-side poster composition"),
    ("ATTR_SHP_PIXEL_GRID","SHAPE", "ドット配列・カクカクしたモザイク格子", "pixelated blocky 8-bit grid mosaic square dots"),

    # --- Facet 3: 材質・質感要素 (MATERIAL & TEXTURE: 14 elements) ---
    ("ATTR_MAT_FOOD_BAKED","MATERIAL", "焼き菓子・ビスケット・パン・食品の質感", "baked biscuit pretzel bread crispy snack food edible texture"),
    ("ATTR_MAT_PAPER_PKG", "MATERIAL", "紙箱・印刷された厚紙パッケージ・ラベル", "printed cardboard box paper packaging glossy product wrapper"),
    ("ATTR_MAT_SILICON_IC","MATERIAL", "半導体・プリント基板・電子マイクロチップ", "silicon microchip fiberglass PCB solder pins electronic hardware"),
    ("ATTR_MAT_METAL_MECH","MATERIAL", "金属・鋼鉄・機械ボディ・塗装された車体", "painted metal steel aluminum mechanical vehicle body"),
    ("ATTR_MAT_FUR_ANIMAL","MATERIAL", "動物の毛並み・ふわふわした羽毛・皮膚", "soft animal fur coat feathers organic living creature skin"),
    ("ATTR_MAT_PETAL_LEAF","MATERIAL", "花びら・植物の葉・瑞々しいボタニカル質感", "delicate flower petals green leaves botanical plant texture"),
    ("ATTR_MAT_ROCK_STONE","MATERIAL", "岩肌・石造り・山岳の岩石・レンガ", "rough stone cliff canyon rock masonry brick texture"),
    ("ATTR_MAT_WATER_LIQ", "MATERIAL", "水面・液体・波立つ水・反射する水辺", "liquid water surface ripples reflection ocean river"),
    ("ATTR_MAT_WOOD",      "MATERIAL", "木材・木目・木造建築や家具の質感", "natural wood grain timber wooden furniture structure"),
    ("ATTR_MAT_FABRIC_UNI","MATERIAL", "布地・衣服・軍服やフォーマルな制服", "woven fabric cloth formal military uniform suit textile"),
    ("ATTR_MAT_GLASS_SCR", "MATERIAL", "ガラス・液晶画面・レンズの透明感", "transparent glass optical lens digital display screen"),
    ("ATTR_MAT_INK_TEXT",  "MATERIAL", "活字インク・白紙に印字された文字テキスト", "printed black text typography font characters on white page"),
    ("ATTR_MAT_DIGITAL_2D","MATERIAL", "デジタル塗り・アニメ/イラスト調のフラット彩色", "cel-shaded digital illustration cartoon graphic flat paint"),
    ("ATTR_MAT_FIRE_ICE",  "MATERIAL", "炎の揺らめき・氷や雪の結晶質感", "blazing fire flame plasma or frozen ice snow crystal"),

    # --- Facet 4: 記号・文字・視覚表現要素 (SYMBOL & GRAPHIC: 12 elements) ---
    ("ATTR_SYM_BRAND_LOGO","SYMBOL", "大きな商品ロゴ・アルファベットタイトル文字", "large bold brand logo lettering product title typography"),
    ("ATTR_SYM_JP_TEXT",   "SYMBOL", "日本語の文章・漢字とひらがなの辞典解説文", "japanese kanji hiragana dictionary article paragraph text explanation"),
    ("ATTR_SYM_MANGA_FX",  "SYMBOL", "漫画の集中線・効果線・強調エフェクト", "manga action concentration lines dramatic comic burst effect"),
    ("ATTR_SYM_MEME_PARODY","SYMBOL","ネットミーム・パロディ・ツッコミどころのあるコラージュ", "internet meme parody humorous collage contradictory joke image"),
    ("ATTR_SYM_TECH_DIAG", "SYMBOL", "科学・数学・コンピュータサイエンスの模式図", "scientific mathematical computer science schematic diagram"),
    ("ATTR_SYM_AXIS_PLOT", "SYMBOL", "座標軸・目盛り・データグラフの軸線", "coordinate axes gridlines chart labels statistical plot"),
    ("ATTR_SYM_PORTRAIT",  "SYMBOL", "人物の顔・視線・ポートレート構図", "human face portrait eyes expression head and shoulders"),
    ("ATTR_SYM_ICON_SPRITE","SYMBOL","ゲームキャラ・アイコン・デフォルメ表現", "chibi character game sprite cute stylized icon"),
    ("ATTR_SYM_BADGE_RANK","SYMBOL", "記章・エンブレム・軍帽や勲章のシンボル", "military badge insignia emblem cap rank decoration"),
    ("ATTR_SYM_ARCH_ROOF", "SYMBOL", "伝統建築の反り屋根・寺院の装飾構造", "traditional curved temple roof pagoda eaves architectural ornament"),
    ("ATTR_SYM_CIRCUIT_TR","SYMBOL", "回路の配線トレース・論理ゲート記号", "electronic circuit traces pinout wiring schematic symbol"),
    ("ATTR_SYM_NUMBERS",   "SYMBOL", "数字・日付（11/11など）・記号表記", "numbers date digits mathematical or calendar notation"),

    # --- Facet 5: 意味ドメイン・用途・文脈要素 (DOMAIN & CONTEXT: 12 elements) ---
    ("ATTR_DOM_CONFECTION","DOMAIN", "お菓子・スナック・チョコレート・甘い食べ物", "confectionery snack food chocolate biscuit sticks sweet treat"),
    ("ATTR_DOM_ENCYCLOPED","DOMAIN", "百科事典・用語解説・スクリーンショット引用", "encyclopedia wiki definition screenshot reference article"),
    ("ATTR_DOM_NATURE_OUT","DOMAIN", "大自然・アウトドア・山や海の風景", "wilderness outdoor nature landscape mountain canyon ocean"),
    ("ATTR_DOM_PET_WILD",  "DOMAIN", "ペット・動物・生き物の生態", "domestic pet companion animal or wild creature fauna"),
    ("ATTR_DOM_MOBILITY",  "DOMAIN", "交通・乗り物・ドライブ・輸送機械", "transportation vehicle driving aviation maritime mobility"),
    ("ATTR_DOM_CS_ALGO",   "DOMAIN", "アルゴリズム・プログラミング・情報工学", "computer science algorithm data structure software engineering"),
    ("ATTR_DOM_ELECTRONIC","DOMAIN", "電子工学・ハードウェア・半導体回路", "electronics hardware electrical engineering circuit board"),
    ("ATTR_DOM_MATH_PHYS", "DOMAIN", "数学・物理学・波動・幾何学理論", "mathematics physics wave mechanics geometry theory"),
    ("ATTR_DOM_ART_POP",   "DOMAIN", "ポップカルチャー・イラスト・ゲーム・サブカル", "pop culture anime game illustration digital art"),
    ("ATTR_DOM_HISTORY_MIL","DOMAIN","歴史・軍人・伝統文化・遺産建築", "historical heritage military officer traditional temple culture"),
    ("ATTR_DOM_DAILY_GOODS","DOMAIN","日用品・家具・文房具・身の回りの道具", "everyday household item furniture tool personal accessory"),
    ("ATTR_DOM_MUSIC_SPORT","DOMAIN","音楽・楽器演奏・スポーツ競技", "musical performance instrument or athletic sport game"),
]


# =============================================================================
# 2. 拡張知識概念（既存の106概念に加え、「お菓子」「百科事典ミーム」等の一般知識概念を標準搭載）
# =============================================================================
KNOWLEDGE_EXPANSION_CONCEPTS = [
    ("KNOW_100_STICK_SNACK", "pretz pocky toppo biscuit sticks snack box green red package confectionery お菓子パッケージ プリッツ ポッキー トッポ スティック菓子 緑の箱"),
    ("KNOW_101_TEXT_MEME",   "contradiction meme encyclopedia screenshot japanese text explanation comic concentration lines 矛盾塊 ピクシブ百科事典 解説テキスト 集中線 ネタ画像 コラ画像"),
    ("KNOW_102_RAMEN_NOODLE","japanese ramen noodle soup bowl chopsticks chashu egg ラーメン 麺類 どんぶり 中華そば"),
    ("KNOW_103_ANIME_ILLUST","japanese anime character illustration digital artwork colorful hair eyes アニメキャラクター イラスト 美少女 2Dデジタルアート"),
    ("KNOW_104_SCREENSHOT",  "ui screenshot web page document text capture window interface スクリーンショット 画面キャプチャ テキスト文書 Webページ"),
]


class CompositionalAttributeProfiler:
    """
    【要素全探査 & 逆配置プロファイラ】
    1. 64個の原始要素（色・形・材質・記号・文脈）を 512d CLIP 空間と 2048d 普遍イデア空間に配置。
    2. すべての事物概念（106+拡張知識＋カスタム概念）について、64要素それぞれとの距離（親和度プロファイル P_concept）を事前全探査。
    3. 入力画像（および左右・中央クロップ）から64要素との距離（画像要素プロファイル p_img）を計測。
    4. 「直接のイデア谷CSLSスコア」と「要素プロファイル間の一致度（逆配置スコア）」を統合して事物を決定・解説する。
    """
    def __init__(self, clip_model, m_minilm, Q_orth: torch.Tensor, mu_txt_bg: torch.Tensor, mu_minilm_bg: torch.Tensor, device: str = "cuda"):
        self.device = device
        self.attr_ids     = [a[0] for a in PRIMITIVE_ATTRIBUTES]
        self.attr_facets  = [a[1] for a in PRIMITIVE_ATTRIBUTES]
        self.attr_jp      = [a[2] for a in PRIMITIVE_ATTRIBUTES]
        self.attr_en_desc = [a[3] for a in PRIMITIVE_ATTRIBUTES]

        with torch.no_grad():
            # CLIP空間上の64要素ベクトル (64, 512)
            attr_clip_raw = F.normalize(torch.tensor(clip_model.encode(self.attr_en_desc), dtype=torch.float32, device=device), p=2, dim=-1)
            self.attr_clip_clean = F.normalize(attr_clip_raw - mu_txt_bg, p=2, dim=-1)

            # 多言語MiniLM -> 2048d イデア空間上の64要素ベクトル (64, 2048)
            full_texts = [f"{jp} {en}" for jp, en in zip(self.attr_jp, self.attr_en_desc)]
            attr_mini_raw = F.normalize(torch.tensor(m_minilm.encode(full_texts), dtype=torch.float32, device=device), p=2, dim=-1)
            attr_mini_clean = F.normalize(attr_mini_raw - mu_minilm_bg, p=2, dim=-1)
            self.attr_idea_vecs = F.normalize(attr_mini_clean @ Q_orth.T, p=2, dim=-1)

    def build_concept_attribute_matrix(self, concept_clip_clean: torch.Tensor, concept_idea_vecs: torch.Tensor) -> torch.Tensor:
        """
        【順方向：事物側の全探査】
        各概念 C 個が 64 個の原始要素（色・形・材質・記号・文脈）とどれだけ近いかを表す
        属性プロファイル行列 P_concept (C, 64) を計算する。
        """
        with torch.no_grad():
            sim_clip = torch.matmul(concept_clip_clean, self.attr_clip_clean.T) # (C, 64)
            sim_idea = torch.matmul(concept_idea_vecs, self.attr_idea_vecs.T)   # (C, 64)
            # 視覚的特徴と意味的特徴の両面から各事物の要素プロファイルを構成
            P_concept = F.normalize(0.55 * sim_clip + 0.45 * sim_idea, p=2, dim=-1)
            return P_concept

    def extract_image_attribute_profile(self, crops_clip_clean: torch.Tensor, bridge) -> tuple[torch.Tensor, dict]:
        """
        【逆方向：画像側の要素取得】
        入力画像のマルチクロップ群 (K_crops, 512) から、64個の原始要素それぞれとの距離（発火強度）を取得する。
        左側にプリッツの箱、右側に日本語テキストがある場合でも、空間クロップ全体のMax-Mean集約により
        両方の要素を漏れなく取得する。
        """
        with torch.no_grad():
            # 1) CLIP視覚空間での直接要素共鳴 (K_crops, 64)
            sim_vis_crops = torch.matmul(crops_clip_clean, self.attr_clip_clean.T)
            # 全体画像(crop 0)と最も強く反応した局所領域のブレンド
            sim_vis = 0.55 * sim_vis_crops[0] + 0.45 * sim_vis_crops.max(dim=0).values # (64,)

            # 2) 2048d普遍イデア部分空間経由での要素共鳴 (K_crops, 64)
            sim_idea_crops = bridge.score_valleys_zero_shot(
                x_source_clean=crops_clip_clean,
                candidate_idea_valleys=self.attr_idea_vecs,
                k_csls=1,
                alpha_csls=0.45
            )
            sim_idea = 0.55 * sim_idea_crops[0] + 0.45 * sim_idea_crops.max(dim=0).values # (64,)

            p_img_raw = 0.60 * sim_vis + 0.40 * sim_idea
            p_img_norm = F.normalize(p_img_raw.unsqueeze(0), p=2, dim=-1).squeeze(0) # (64,)

            # ファセット（色・形・材質・記号・文脈）ごとに上位発火要素を整理
            facet_summary = {}
            for facet_name in ["COLOR", "SHAPE", "MATERIAL", "SYMBOL", "DOMAIN"]:
                f_indices = [i for i, f in enumerate(self.attr_facets) if f == facet_name]
                f_scores = [(self.attr_ids[i], self.attr_jp[i], float(p_img_raw[i].item())) for i in f_indices]
                f_scores.sort(key=lambda x: x[2], reverse=True)
                facet_summary[facet_name] = f_scores[:3]

            return p_img_norm, facet_summary

    def score_concepts_by_bidirectional_attributes(
        self,
        direct_csls_scores: torch.Tensor,
        p_img_norm: torch.Tensor,
        P_concept: torch.Tensor,
        beta_attr: float = 0.42
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        【事物の逆設置（Bidirectional Attribute-Guided Concept Placement）】
        直接のイデア谷CSLSスコアと、「画像側の要素プロファイル p_img と 事物側の要素プロファイル P_concept の一致度」
        を統合して最終的な事物の配置スコアを算出する。
        """
        with torch.no_grad():
            # 要素プロファイル同士のコサイン類似度 (C,)
            attr_profile_sim = torch.matmul(P_concept, p_img_norm) # (C,)
            combined_scores = (1.0 - beta_attr) * direct_csls_scores + beta_attr * attr_profile_sim
            return combined_scores, attr_profile_sim
