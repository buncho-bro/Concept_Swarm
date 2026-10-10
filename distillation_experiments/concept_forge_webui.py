"""
【ファイル概要】
concept_forge_webui.py

引力・反発のポテンシャル地形をGUI上で視覚的に調整し、リアルタイムにモデルの思考を操作できるコンセプトフォージのWebUI。
"""

import gradio as gr
import os
import gc
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import LoraConfig, get_peft_model, PeftModel
from threading import Thread
import json
from safetensors import safe_open
from huggingface_hub import snapshot_download

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shallow_extractor import ShallowConceptExtractor, InterferenceConfig

# --- グローバル状態（チャット・ベンチマーク用） ---
current_model = None
current_tokenizer = None
current_model_name = "None"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 純度100%の概念アンカー（Universal Language / English） ---
IDEA_PRESETS = {
    "Universal (General Knowledge)": [
        "Time", "Space", "Existence", "Nothingness", "Truth", "Logic", "Philosophy", "Ethics", "Morality", "Justice", "Freedom", "Equality", "Peace", "War", "History", "Future", "Past", "Eternity",
        "Mathematics", "Chemistry", "Biology", "Geometry", "Algebra", "Probability", "Statistics", "Quantum", "Particle", "Light", "Gravity", "Magnetism", "Electricity", "Energy", "Mass", "Velocity",
        "Economy", "Market", "Capital", "Currency", "Labor", "Industry", "Trade", "Enterprise", "Society", "Culture", "Law", "Nation", "Democracy", "Dictatorship", "Revolution", "Rights",
        "Art", "Music", "Painting", "Sculpture", "Literature", "Poetry", "Theater", "Movie", "Beauty", "Expression", "Creation", "Tradition", "Religion", "Faith", "God", "Spirit", "Soul",
        "Plant", "Tree", "Flower", "Grass", "Forest", "Ocean", "Mountain", "River", "Desert", "Climate", "Weather", "Rain", "Snow", "Wind", "Thunder", "Earth", "Planet", "Star",
        "Machine", "Tool", "Engine", "Robot", "Circuit", "Communication", "Metal", "Iron", "Gold", "Silver", "Copper", "Water", "Fire", "Earth_soil", "Air", "Oxygen", "Hydrogen",
        "Joy", "Sadness", "Fear", "Surprise", "Love", "Hate", "Hope", "Despair", "Loneliness", "Happiness", "Pain", "Pleasure", "Memory", "Thought", "Consciousness", "Unconscious"
    ],
    "Coding & Logic (Programming)": [
        "Function", "Variable", "Loop", "Recursion", "Branch", "Memory", "Type", "Object", "Compile", "Bug", "Algorithm", "Pointer", "Concurrency", "Asynchronous", "Stack", "Queue",
        "Interface", "Abstract", "Inheritance", "Polymorphism", "Database", "Query", "Index", "Network", "Protocol", "Security", "Encryption", "Authentication", "Authorization", "Session",
        "Thread", "Process", "Lock", "Mutex", "Semaphore", "Cache", "Buffer", "Stream", "Socket", "API", "Framework", "Library", "Module", "Package", "Deployment", "Container",
        "Virtualization", "Cloud", "Server", "Client", "Frontend", "Backend", "Middleware", "Node", "Cluster", "Scalability", "Latency", "Bandwidth", "Throughput", "Optimization",
        "Graph", "Tree", "Hash", "Array", "List", "Dictionary", "Tuple", "String", "Integer", "Float", "Boolean", "Null", "Void", "Exception", "Error", "Warning", "Log", "Debug",
        "Test", "Assertion", "Mock", "Stub", "Version", "Commit", "Merge", "Branch", "Repository", "Syntax", "Semantics", "Parser", "Lexer", "Interpreter", "Compiler"
    ],
    "Math & Reasoning (Abstract Logic)": [
        "Number", "Infinity", "Proof", "Theorem", "Probability", "Set", "Dimension", "Derivative", "Integral", "Matrix", "Topology", "Symmetry", "Limit", "Hypothesis", "Deduction",
        "Induction", "Contradiction", "Truth", "Falsehood", "Axiom", "Lemma", "Corollary", "Equation", "Inequality", "Function", "Variable", "Constant", "Coefficient", "Exponent",
        "Logarithm", "Root", "Fraction", "Decimal", "Vector", "Tensor", "Scalar", "Space", "Curve", "Surface", "Volume", "Angle", "Distance", "Area", "Perimeter", "Radius",
        "Diameter", "Circumference", "Polygon", "Polyhedron", "Sphere", "Cylinder", "Cone", "Pyramid", "Graph", "Node", "Edge", "Path", "Cycle", "Tree", "Forest", "Network",
        "Distribution", "Mean", "Median", "Mode", "Variance", "Deviation", "Correlation", "Regression", "Sample", "Population", "Confidence", "Significance", "Algorithm", "Complexity"
    ],
    "Empathy & Emotion (Social & RP)": [
        "Joy", "Sadness", "Anger", "Empathy", "Comfort", "Humor", "Joke", "Irony", "Gratitude", "Apology", "Encouragement", "Bond", "Family", "Friend", "Love", "Worry", "Relief",
        "Anxiety", "Hope", "Despair", "Loneliness", "Happiness", "Pain", "Pleasure", "Memory", "Thought", "Consciousness", "Unconscious", "Dream", "Nightmare", "Fantasy",
        "Reality", "Illusion", "Delusion", "Trust", "Betrayal", "Loyalty", "Treason", "Honesty", "Deceit", "Truth", "Lie", "Secret", "Confession", "Forgiveness", "Revenge",
        "Justice", "Injustice", "Fairness", "Unfairness", "Equality", "Inequality", "Freedom", "Oppression", "Peace", "Conflict", "Harmony", "Discord", "Cooperation", "Competition",
        "Sympathy", "Antipathy", "Apathy", "Passion", "Desire", "Lust", "Greed", "Envy", "Pride", "Humility", "Courage", "Cowardice", "Bravery", "Fear", "Terror", "Horror"
    ]
}

CONCEPT_WORDS = IDEA_PRESETS["Universal (General Knowledge)"] # Default fallback for chat
def get_preset_text(preset_name):
    return ", ".join(IDEA_PRESETS[preset_name])

analogy_tests = [("King", "Man", "Woman", "Queen"), ("Tokyo", "Japan", "France", "Paris"), ("Doctor", "Hospital", "School", "Teacher"), ("Car", "Road", "Track", "Train"), ("Day", "Sun", "Night", "Moon")]
hierarchy_tests = [("Animal", "Dog", "Desk"), ("Science", "Physics", "Apple"), ("Emotion", "Anger", "Stone"), ("Computer", "Software", "Nature"), ("Universe", "Galaxy", "Politics")]

def free_vram():
    global current_model, current_tokenizer, current_model_name
    if current_model is not None:
        del current_model
        current_model = None
    if current_tokenizer is not None:
        del current_tokenizer
        current_tokenizer = None
    current_model_name = "None"
    gc.collect()
    torch.cuda.empty_cache()

def get_word_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, -1, :] # Fix: Use last token
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

# ==========================================
# 1. 蒸留ロジック (KNN Graph Distillation)
# ==========================================
def distill_process(teacher_id, student_id, save_dir, r_value, epochs_val, knn_k, custom_concepts_text, algo_type, low_vram_mode, shallow_mode, cache_dir_val):
    concept_words = [w.strip() for w in custom_concepts_text.split(",") if w.strip()]
    if len(concept_words) < 10:
        yield "Error: At least 10 concept words are required for stable topology."
        return
        
    cache_kwargs = {}
    if cache_dir_val and cache_dir_val.strip():
        os.makedirs(cache_dir_val.strip(), exist_ok=True)
        os.environ["HF_HOME"] = cache_dir_val.strip()
        cache_kwargs = {"cache_dir": cache_dir_val.strip()}
        
    # 自動フォルダ命名ロジック (デフォルトのままであれば固有のフォルダ名を生成)
    if save_dir == "./concept_lora_output":
        safe_t = teacher_id.split('/')[-1].replace('-', '_')
        safe_s = student_id.split('/')[-1].replace('-', '_')
        if "KNN" in algo_type:
            actual_save_dir = f"./lora_{safe_t}_to_{safe_s}_K{int(knn_k)}"
        else:
            actual_save_dir = f"./lora_{safe_t}_to_{safe_s}_Topological"
    else:
        actual_save_dir = save_dir
        
    log_text = f"=== Starting Distillation ({algo_type}) ===\n"
    log_text += f"Target Directory: {actual_save_dir}\n"
    if low_vram_mode:
        log_text += "🐢 Extreme Low VRAM Mode ENABLED (Layer-by-Layer Offloading)\n"
    if shallow_mode:
        log_text += "⚡ CPU-Only Shallow Extraction ENABLED (Skipping Forward Pass)\n"
    if cache_kwargs:
        log_text += f"💾 Custom Cache Directory: {cache_dir_val.strip()}\n"
    yield log_text
    
    try:
        free_vram()
        os.makedirs(actual_save_dir, exist_ok=True)
        
        if low_vram_mode or shallow_mode:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

        log_text += f"\n[1/3] Loading Teacher ({teacher_id})...\n"
        yield log_text
        teacher_tok = AutoTokenizer.from_pretrained(teacher_id, **cache_kwargs)
        
        teacher_tok = AutoTokenizer.from_pretrained(teacher_id, **cache_kwargs)
        teacher_tok.pad_token = teacher_tok.eos_token
        
        if shallow_mode:
            log_text += f"\n[1/3] ⚡ ULTIMATE Shallow Mode (Great Circle Interference): Extracting Concept Cores directly from disk...\n"
            yield log_text
            
            if cache_dir_val and cache_dir_val.strip():
                os.environ["SHALLOW_EXTRACTOR_CACHE"] = cache_dir_val.strip()
                
            # Import newly created Great Circle Extractor
            from shallow_extractor import ShallowConceptExtractor, InterferenceConfig
            import numpy as np
            
            log_text += f"Initializing ShallowConceptExtractor for {teacher_id}...\n"
            yield log_text
            
            ex = ShallowConceptExtractor(model_id=teacher_id, verbose=False)
            
            log_text += f"Extracting {len(concept_words)} Concept Cores...\n"
            yield log_text
            
            teacher_vecs_list = []
            for i, word in enumerate(concept_words):
                # Extract the primary concept core using Great Circle Interference
                res = ex.extract(word, config=InterferenceConfig(k_max=1))
                teacher_vecs_list.append(res.primary)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(concept_words):
                    yield log_text + f"  -> Processed {i + 1}/{len(concept_words)} words\n"
                    
            teacher_vecs = torch.tensor(np.array(teacher_vecs_list), dtype=torch.float32).to(device)
            # Normalization is naturally handled by the Concept Core
            
            # Clean up
            del ex
            gc.collect()
            
        else:
            if low_vram_mode:
                import psutil
                ram_gb = psutil.virtual_memory().total / (1024**3)
                cpu_mem = f"{int(ram_gb * 0.8)}GiB"
                max_memory = {0: "4GiB", "cpu": cpu_mem}
                offload_dir = "./offload_weights"
                os.makedirs(offload_dir, exist_ok=True)
                log_text += f"Configuring Offload: GPU max 4GB, CPU max {cpu_mem}\n"
                yield log_text
                teacher = AutoModelForCausalLM.from_pretrained(
                    teacher_id, 
                    quantization_config=bnb_config, 
                    device_map="auto", 
                    max_memory=max_memory,
                    offload_folder=offload_dir,
                    **cache_kwargs
                )
            else:
                teacher = AutoModelForCausalLM.from_pretrained(teacher_id, quantization_config=bnb_config, device_map={"": "cuda:0"}, **cache_kwargs)
            teacher.eval()
            
            log_text += f"Extracting Pure Concept Coordinates ({len(concept_words)} words) with Batching...\n"
            yield log_text
            inputs = teacher_tok(concept_words, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = teacher(**inputs, output_hidden_states=True)
                seq_lens = inputs.attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(len(concept_words), device=device)
                teacher_vecs = outputs.hidden_states[-1][batch_indices, seq_lens, :].to(torch.float32)
                teacher_vecs = F.normalize(teacher_vecs, p=2, dim=1)
                
            del teacher
            del outputs
        
        P_T_full = None
        normalized_T_dist = None
        
        if "KNN" in algo_type:
            log_text += "Creating KNN Graph (Teacher's Constellation)...\n"
            yield log_text
            
            # Teacherのペアワイズ行列 [N, N]
            T_sim_full = torch.matmul(teacher_vecs, teacher_vecs.T)
            
            K = min(int(knn_k), len(concept_words) - 1)
            log_text += f"Using K={K} for KNN Graph extraction...\n"
            yield log_text
            
            temperature = 0.05
            topk_values, topk_indices = torch.topk(T_sim_full, k=min(K+1, T_sim_full.size(-1)), dim=-1)
            
            mask_full = torch.zeros_like(T_sim_full, dtype=torch.bool)
            mask_full.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_values, dtype=torch.bool))
            
            T_sim_masked = T_sim_full.clone()
            T_sim_masked[~mask_full] = -1e9
            P_T_full = F.softmax(T_sim_masked / temperature, dim=-1)
            log_text += "KNN Graph Complete!\n"
            yield log_text
            
        else:
            log_text += "Calculating Topological Scale-Invariant Shape (Global Shape)...\n"
            yield log_text
            
            # 宇宙の重心（穴の中心・大域構造の基準点）
            teacher_centroid = torch.mean(teacher_vecs, dim=0)
            teacher_dists_from_center = torch.norm(teacher_vecs - teacher_centroid, dim=1)
            k_T = torch.max(teacher_dists_from_center) # 宇宙の端までの距離(k_T)
            
            log_text += f"Teacher Scale k(T) = {k_T.item():.4f}\n"
            yield log_text
            
            # ユークリッド距離行列の構築とk(T)による正規化
            T_dist_matrix = torch.cdist(teacher_vecs, teacher_vecs, p=2)
            normalized_T_dist = T_dist_matrix / (k_T + 1e-8)
            log_text += "Topological Normalization Complete!\n"
            yield log_text
        
        log_text += "Cleaning up Teacher memory...\n"
        yield log_text
        try:
            del teacher
        except NameError:
            pass
        del teacher_tok
        del teacher_vecs
        if "KNN" in algo_type:
            del T_sim_full
        gc.collect()
        torch.cuda.empty_cache()
        
        log_text += f"\n[2/3] Loading Student ({student_id}) and setting up LoRA...\n"
        yield log_text
        student_tok = AutoTokenizer.from_pretrained(student_id, **cache_kwargs)
        student_tok.pad_token = student_tok.eos_token
        
        # StudentはVRAM(GPU)上にロードするため、オフロードフラグを持たないクリーンな設定を使う
        student_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(student_id, quantization_config=student_bnb_config, device_map={"": "cuda:0"}, **cache_kwargs)
        
        lora_config = LoraConfig(
            r=int(r_value),
            lora_alpha=int(r_value)*2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        student = get_peft_model(model, lora_config)
        
        log_text += f"\n[3/3] Starting Distillation ({algo_type}) with Full-Batching...\n"
        yield log_text
        optimizer = optim.Adam(student.parameters(), lr=1e-4)
        epochs = int(epochs_val)
        
        # 学習ループ前の事前準備（トークナイズと初期座標の取得）
        inputs = student_tok(concept_words, return_tensors="pt", padding=True, truncation=True).to(device)
        seq_lens = inputs.attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(concept_words), device=device)
        
        # アンカーペナルティのための初期位置（エポック0の形）を保存
        student.eval()
        with torch.no_grad():
            initial_outputs = student(**inputs, output_hidden_states=True)
            initial_vecs = initial_outputs.hidden_states[-1][batch_indices, seq_lens, :].to(torch.float32)
            initial_vecs = F.normalize(initial_vecs, p=2, dim=1).detach()
        
        student.train()
        for epoch in range(epochs):
            total_loss = 0
            
            optimizer.zero_grad()
            
            # 【一撃必殺】バッチサイズ＝全単語 で1回だけForward
            outputs = student(**inputs, output_hidden_states=True)
            
            # 全単語の最終トークンの隠れ状態を抽出 [N, Hidden_Dim]
            current_vecs = outputs.hidden_states[-1][batch_indices, seq_lens, :].to(torch.float32)
            current_vecs_norm = F.normalize(current_vecs, p=2, dim=1)
            
            if "KNN" in algo_type:
                # ペアワイズ類似度行列 [N, N]
                M_S_full = torch.matmul(current_vecs_norm, current_vecs_norm.T)
                
                # Teacherと同じマスク（ネットワークの線）だけを学習
                M_S_masked = M_S_full.clone()
                M_S_masked[~mask_full] = -1e9
                P_S_full = F.softmax(M_S_masked / temperature, dim=-1)
                
                # KL Divergence
                loss = F.kl_div(torch.log(P_S_full + 1e-8), P_T_full, reduction='batchmean')
                
            else:
                # Topological Scale-Invariant Loss (Distance Matrix MSE)
                student_centroid = torch.mean(current_vecs_norm, dim=0)
                student_dists = torch.norm(current_vecs_norm - student_centroid, dim=1)
                k_S = torch.max(student_dists)
                
                # 距離行列 [N, N]
                S_dist_matrix = torch.cdist(current_vecs_norm, current_vecs_norm, p=2)
                normalized_S_dist = S_dist_matrix / (k_S + 1e-8)
                
                # 次元を超えた同相写像の誤差 (形のコピー)
                topology_loss = F.mse_loss(normalized_S_dist, normalized_T_dist)
                
                # 星座全体の回転・反転を防ぐためのアンカーペナルティ
                # 常に「学習開始時」の絶対座標をアンカーにして空間のアライメントを維持
                anchor_penalty = F.mse_loss(current_vecs_norm, initial_vecs)
                
                # 形の形成を主目的としつつ、10%の力で元の位置に繋ぎ止める
                loss = topology_loss + 0.1 * anchor_penalty
            
            loss.backward()
            optimizer.step()
            
            total_loss = loss.item()
                
            # バッチサイズ=全単語となったため、total_lossをそのまま表示
            log_text += f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.5f}\n"
            yield log_text
            
        log_text += f"\n=== Saving LoRA to: {actual_save_dir} ===\n"
        yield log_text
        student.save_pretrained(actual_save_dir)
        
        log_text += f"Distillation Complete! You can now load the model from '{actual_save_dir}' in the Chat/Benchmark tab.\n"
        yield log_text
        
        free_vram()
        
    except Exception as e:
        log_text += f"\nERROR: {str(e)}\n"
        yield log_text

# ==========================================
# 2. モデルロード（チャット/ベンチマーク用）
# ==========================================
def load_model_for_chat(base_id, lora_dir, use_lora, cache_dir_val):
    global current_model, current_tokenizer, current_model_name
    free_vram()
    
    cache_kwargs = {}
    if cache_dir_val and cache_dir_val.strip():
        os.makedirs(cache_dir_val.strip(), exist_ok=True)
        os.environ["HF_HOME"] = cache_dir_val.strip()
        cache_kwargs = {"cache_dir": cache_dir_val.strip()}
    
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        current_tokenizer = AutoTokenizer.from_pretrained(base_id, **cache_kwargs)
        current_model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb_config, device_map={"": "cuda:0"}, **cache_kwargs)
        
        if lora_dir == "None":
            use_lora = False
            
        if use_lora:
            if os.path.exists(lora_dir):
                try:
                    current_model = PeftModel.from_pretrained(current_model, lora_dir)
                    current_model_name = f"Distilled Student (LoRA: {os.path.basename(lora_dir)})"
                except Exception as lora_e:
                    current_model_name = "Base Student (LoRA Load Failed)"
                    return f"❌ ERROR: LoRAの適用に失敗しました。\nベースモデル '{base_id}' と LoRA '{lora_dir}' の次元やアーキテクチャが異なっている可能性があります。\n詳細: {str(lora_e)}"
            else:
                current_model_name = "Base Student"
                return f"⚠️ WARNING: LoRA path '{lora_dir}' not found. Loaded Base Student instead."
        else:
            current_model_name = "Base Student"
            
        return f"Successfully loaded: {current_model_name}"
    except Exception as e:
        return f"Error loading Base Model: {str(e)}"

# ==========================================
# 3. チャットロジック (Deepthink 搭載)
# ==========================================
def chat_fn(message, history, use_deepthink, use_lightweight):
    global current_model, current_tokenizer
    if current_model is None:
        yield "Please load a model first in the options above."
        return
        
    messages = []
    # Fix: Gradio 6 のディクショナリ型やリスト型（マルチモーダル履歴）からテキストのみを安全に抽出
    for msg in history:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, (list, tuple)):
                # 画像などのマルチモーダルデータが含まれる場合、最初のテキスト要素を取得
                content = str(content[0]) if len(content) > 0 else ""
            elif not isinstance(content, str):
                content = str(content)
            messages.append({"role": msg.get("role", "user"), "content": content})
        else:
            # 古いGradioフォーマット
            u_content = str(msg[0]) if not isinstance(msg[0], str) else msg[0]
            a_content = str(msg[1]) if not isinstance(msg[1], str) else msg[1]
            messages.append({"role": "user", "content": u_content})
            messages.append({"role": "assistant", "content": a_content})
            
    if not isinstance(message, str):
        message = str(message)
        
    system_prompt = ""
    thinking_log = ""
    
    # --- Deepthink (Fractal Routing / Local Gravity Cancellation) Mode ---
    if use_deepthink:
        # 1. ユーザーベクトル (U) の抽出
        inputs_dt = current_tokenizer(message[-15:], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_dt = current_model(**inputs_dt, output_hidden_states=True)
        user_vec = F.normalize(outputs_dt.hidden_states[-1][0, -1, :].to(torch.float32), p=2, dim=0)
        
        # 2. 概念辞書のベクトル化（キャッシュを使用して超高速化）
        global cached_concept_vecs, cached_model_id
        if 'cached_model_id' not in globals() or cached_model_id != current_model_name or cached_concept_vecs is None:
            yield "Initializing Concept Topography Cache (First time only...)\n"
            concept_vecs = []
            for w in CONCEPT_WORDS:
                concept_vecs.append(get_word_embedding(current_model, current_tokenizer, w))
            cached_concept_vecs = torch.stack(concept_vecs)
            cached_model_id = current_model_name
            
        concept_vecs = cached_concept_vecs
        
        # 3. Macro Routing: 最も近い「巨大な谷の底 (C_macro)」を見つける
        sims_macro = F.cosine_similarity(user_vec.unsqueeze(0), concept_vecs)
        macro_idx = torch.argmax(sims_macro).item()
        c_macro_vec = concept_vecs[macro_idx]
        macro_concept = CONCEPT_WORDS[macro_idx]
        
        # 3. Local Gravity Cancellation (谷を開く・重力をなくす)
        # 直交射影を用いて、巨大な谷の重力成分を空間全体から消し去る
        
        # ユーザーベクトルの重力除去と突起の正規化(拡大)
        dot_u = torch.dot(user_vec, c_macro_vec)
        u_micro = user_vec - dot_u * c_macro_vec
        u_micro = F.normalize(u_micro, p=2, dim=0)
        
        # すべての概念ベクトルの重力除去と突起の正規化(拡大)
        dot_c = torch.matmul(concept_vecs, c_macro_vec) 
        concept_micro_vecs = concept_vecs - dot_c.unsqueeze(1) * c_macro_vec.unsqueeze(0)
        concept_micro_vecs = F.normalize(concept_micro_vecs, p=2, dim=1)
        
        # 4. Micro Routing: 拡大されたフラクタル空間での精緻な探索 (突起の演算)
        sims_micro = F.cosine_similarity(u_micro.unsqueeze(0), concept_micro_vecs)
        
        # 自分自身 (macro_concept) は残差が0になり計算不能になるため除外
        sims_micro[macro_idx] = -1.0 
        
        # sims_micro を高い順にソート
        sorted_sims, sorted_idx = torch.sort(sims_micro, descending=True)
        
        routed_concepts = []
        max_sim = sorted_sims[0].item()
        
        # 動的思考打ち切り (Adaptive Stopping / Drop-off detection)
        # 最大5手まで思考。最大の類似度から「0.15」以上ガクッと下がったら無関係と判断して打ち切る
        for i in range(5):
            current_sim = sorted_sims[i].item()
            # 崖の検知、または類似度自体が0以下（全く無関係）ならストップ
            if (max_sim - current_sim) > 0.15 or current_sim <= 0.0:
                break
            routed_concepts.append(CONCEPT_WORDS[sorted_idx[i].item()])
            
        # 安全装置: 最低でも1つは概念を確保
        if not routed_concepts:
            routed_concepts.append(CONCEPT_WORDS[sorted_idx[0].item()])
        
        thinking_log = f"**[Fractal Deepthink]**\n"
        thinking_log += f"Macro Valley (重力源): `{macro_concept}`\n"
        thinking_log += f"Micro Routing (重力除去後の突起探索): `{'` -> `'.join(routed_concepts)}`\n\n"
        yield thinking_log
        
        # 抽出した概念をヒントとしてプロンプトに注入
        # AIに「自己修正権（拒否権）」を与え、泥沼化を防ぐ
        system_prompt = (
            f"System: [Latent Trajectory Suggestion]\n"
            f"- Macro Concept: '{macro_concept}'\n"
            f"- Micro Concepts: {', '.join(routed_concepts)}\n\n"
            f"Instruction: Use these concepts as a creative scaffolding for your reasoning. "
            f"However, you have absolute authority to self-correct: if you determine that any of these concepts lead to a logical dead end or are irrelevant to the user's actual prompt, you MUST ignore them and dynamically correct your trajectory to provide the most accurate answer.\n\n"
        )

    messages.append({"role": "user", "content": system_prompt + message})
    
    text = current_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = current_tokenizer(text, return_tensors="pt").to(device)
    
    # --- 🍃 Lightweight Mode (FAISS 100-words Decoding) ---
    if use_lightweight:
        import faiss
        import numpy as np
        
        global cached_faiss_index, cached_lm_head_weight, cached_faiss_model_id
        
        # 15万語のFAISSインデックスの構築（初回のみキャッシュ）
        if 'cached_faiss_model_id' not in globals() or cached_faiss_model_id != current_model_name or cached_faiss_index is None:
            yield thinking_log + "🍃 Initializing Lightweight FAISS Index (150,000 words). First time only...\n"
            cached_lm_head_weight = current_model.lm_head.weight.detach()
            lm_head_np = cached_lm_head_weight.cpu().to(torch.float32).numpy()
            faiss.normalize_L2(lm_head_np)
            D = lm_head_np.shape[1]
            cached_faiss_index = faiss.IndexFlatIP(D)
            cached_faiss_index.add(lm_head_np)
            cached_faiss_model_id = current_model_name
            
        generated_text = thinking_log
        yield generated_text
        
        # カスタム推論ループ (KVキャッシュ付き)
        past_key_values = None
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = current_model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
        past_key_values = outputs.past_key_values
        hidden = outputs.hidden_states[-1][0, -1, :]
        
        for _ in range(2048):
            # FAISS高速探査
            hidden_np = hidden.cpu().to(torch.float32).numpy().reshape(1, -1)
            faiss.normalize_L2(hidden_np)
            _, I = cached_faiss_index.search(hidden_np, 100) # K=100
            candidates = I[0]
            
            # 局所文法評価
            local_weights = cached_lm_head_weight[candidates]
            local_logits = torch.matmul(hidden.unsqueeze(0), local_weights.T)
            
            best_local_idx = torch.argmax(local_logits, dim=-1).item()
            next_token = candidates[best_local_idx]
            
            # 停止条件 (EOSトークンなど)
            if next_token in [current_tokenizer.eos_token_id, current_tokenizer.pad_token_id] or next_token == 151645: # 151645 is typical im_end for Qwen
                break
                
            new_str = current_tokenizer.decode([next_token], skip_special_tokens=True)
            generated_text += new_str
            yield generated_text
            
            # 次のステップ
            input_ids = torch.tensor([[next_token]]).to(device)
            with torch.no_grad():
                outputs = current_model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            past_key_values = outputs.past_key_values
            hidden = outputs.hidden_states[-1][0, -1, :]
            
    else:
        # --- 従来方式 (全15万語計算 + Threading) ---
        streamer = TextIteratorStreamer(current_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=2048, # 512 から 2048 に拡張 (日本語の回答途切れ防止)
            do_sample=True,
            temperature=0.7,
        )
        
        thread = Thread(target=current_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        generated_text = thinking_log
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

# ==========================================
# 4. ベンチマークロジック
# ==========================================
def test_multihop_routing(model, tokenizer, start, relations, target):
    current_vec = get_word_embedding(model, tokenizer, start)
    for rel in relations:
        rel_vec = get_word_embedding(model, tokenizer, rel)
        current_vec = F.normalize(current_vec + rel_vec, p=2, dim=0)
    target_vec = get_word_embedding(model, tokenizer, target)
    return F.cosine_similarity(current_vec.unsqueeze(0), target_vec.unsqueeze(0)).item()

def test_logic_chain(model, tokenizer, a, a_is_b, c, target_b):
    vec_a = get_word_embedding(model, tokenizer, a)
    vec_b = get_word_embedding(model, tokenizer, a_is_b)
    vec_c = get_word_embedding(model, tokenizer, c)
    logic_direction = F.normalize(vec_b - vec_a, p=2, dim=0)
    pred_b = F.normalize(vec_c + logic_direction, p=2, dim=0)
    target_vec = get_word_embedding(model, tokenizer, target_b)
    return F.cosine_similarity(pred_b.unsqueeze(0), target_vec.unsqueeze(0)).item()

# ==========================================
# 4. ベンチマークロジック
# ==========================================
def run_benchmark_ui():
    global current_model, current_tokenizer, current_model_name
    if current_model is None:
        return "Please load a model first."
        
    log = f"Running Benchmark for: {current_model_name}...\n\n"
    
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        va, vb, vc, vd = [get_word_embedding(current_model, current_tokenizer, w) for w in (a, b, c, d)]
        v_pred = F.normalize(va - vb + vc, p=2, dim=0)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vd.unsqueeze(0)).item()
        analogy_scores.append(score)
    avg_analogy = np.mean(analogy_scores)
    
    hierarchy_scores = []
    for base, target, outlier in hierarchy_tests:
        v_base, v_tgt, v_out = [get_word_embedding(current_model, current_tokenizer, w) for w in (base, target, outlier)]
        sim_tgt = F.cosine_similarity(v_base.unsqueeze(0), v_tgt.unsqueeze(0)).item()
        sim_out = F.cosine_similarity(v_base.unsqueeze(0), v_out.unsqueeze(0)).item()
        score = sim_tgt - sim_out
        hierarchy_scores.append(score)
    avg_hierarchy = np.mean(hierarchy_scores)
    
    log += f"Analogy (推論力): {avg_analogy:.4f}\n"
    log += f"Hierarchy (階層理解): {avg_hierarchy:.4f}\n"
    
    log += "\n--- Thinking Benchmark (Concept Routing) ---\n"
    hop_tests = [("Japan", ["Tokyo", "Tower"], "Skytree"), ("France", ["Paris", "Tower"], "Eiffel_Tower"), ("Science", ["Apple", "Universal_Gravitation"], "Gravity")]
    hop_scores = [test_multihop_routing(current_model, current_tokenizer, s, r, t) for s, r, t in hop_tests]
    
    logic_tests = [("Bird", "Move", "Penguin", "Move"), ("Mammal", "Animal", "Dog", "Animal"), ("Ice", "Cold", "Water", "Cold")]
    logic_scores = [test_logic_chain(current_model, current_tokenizer, a, b, c, t) for a, b, c, t in logic_tests]
    
    log += f"Multi-hop (Continuous Trajectory): {np.mean(hop_scores):.4f}\n"
    log += f"Logic Chain (Logical Inference): {np.mean(logic_scores):.4f}\n"
    return log


import glob

def get_available_loras():
    lora_dirs = [d for d in glob.glob("./lora_*") if os.path.isdir(d)]
    if os.path.exists("./concept_lora_output"):
        lora_dirs.append("./concept_lora_output")
    return ["None"] + lora_dirs

def refresh_lora_list():
    return gr.Dropdown(choices=get_available_loras())

shallow_ex_instance = None
def get_shallow_extractor():
    global shallow_ex_instance
    if shallow_ex_instance is None:
        shallow_ex_instance = ShallowConceptExtractor(verbose=False)
    return shallow_ex_instance

def extract_shallow_concepts(text):
    if not text.strip():
        return "Please enter some text."
    
    ex = get_shallow_extractor()
    res = ex.extract(text)
    
    html = f"<h3>✅ Extraction completed in {res.elapsed_ms:.1f} ms (Sigma: {res.sigma:.4f} rad)</h3>"
    
    for k, core in enumerate(res.cores):
        html += f"<h4>Concept Core #{k+1}</h4>"
        html += f"<p><b>Potential (Phi):</b> {core.potential:.3f} &nbsp;|&nbsp; <b>Resonance:</b> {core.resonance:.3%}</p>"
        
        tops = core.top_tokens(res.tokens, k=10)
        
        html += "<div style='margin-bottom: 20px;'>"
        for t, w in tops:
            t_clean = t.replace('Ġ', '').strip()
            if not t_clean: continue
            
            intensity_ratio = w / (tops[0][1] + 1e-9)
            color = "white" if intensity_ratio > 0.5 else "black"
            
            html += f"<span style='display:inline-block; margin:4px; padding:6px 12px; background-color:rgba(46, 134, 171, {intensity_ratio}); color:{color}; border-radius:16px; border:1px solid #2E86AB; font-weight:bold;'>{t_clean} <small style='opacity:0.7'>({w:.2f})</small></span>"
        html += "</div><hr>"
        
    return html

# ==========================================
# Gradio UI 構築
# ==========================================

TEACHER_CHOICES = [
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "google/gemma-2-9b",
    "mistralai/Mistral-7B-v0.3"
]

STUDENT_CHOICES = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it"
]

with gr.Blocks(title="Concept Forge", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚗️ Concept Forge (KNN Graph Distillation)")
    gr.Markdown("純粋な概念座標から「最も近い8個の隣人ネットワーク（星座）」だけを抽出し、小規模モデルの脳をパンクさせずに限界突破させるOSSツールです。")
    
    with gr.Row():
        cache_dir_input = gr.Textbox(label="⚙️ Global Settings: HuggingFace Cache Directory (Cドライブの容量不足を防ぐための保存先変更)", placeholder="例: D:\\HuggingFaceCache", value="", interactive=True)
    
    with gr.Tabs():
        # --- TAB 1: 蒸留 ---
        with gr.TabItem("⚙️ Distillation"):
            with gr.Row():
                with gr.Column():
                    t_id = gr.Dropdown(choices=TEACHER_CHOICES, value="Qwen/Qwen2.5-7B", label="Teacher Model (リストから選択 or 手動入力)", allow_custom_value=True)
                    s_id = gr.Dropdown(choices=STUDENT_CHOICES, value="Qwen/Qwen2.5-1.5B-Instruct", label="Student Model (リストから選択 or 手動入力)", allow_custom_value=True)
                    
                    gr.Markdown("### Concept Anchors (イデアの選択)")
                    preset_dropdown = gr.Dropdown(choices=list(IDEA_PRESETS.keys()), value="Universal (General Knowledge)", label="Load Preset")
                    custom_concepts_text = gr.Textbox(value=get_preset_text("Universal (General Knowledge)"), label="Custom Concept Words (カンマ区切りで自由編集可能)", lines=5)
                    preset_dropdown.change(fn=get_preset_text, inputs=[preset_dropdown], outputs=[custom_concepts_text])
                    
                    save_d = gr.Textbox(label="Drive/Save Path for LoRA", value="./concept_lora_output")
                    algo_radio = gr.Radio(choices=["KNN Graph (Local Topology)", "Topological Scale-Invariant (Global Shape)"], value="Topological Scale-Invariant (Global Shape)", label="Distillation Algorithm")
                    low_vram_chk = gr.Checkbox(label="🐢 Extreme Low VRAM Mode (Layer-by-Layer Offloading)", value=False, info="Check this to extract from massive models (e.g., 70B+) using RAM/Disk layer-by-layer offloading.")
                    shallow_chk = gr.Checkbox(label="⚡ CPU-Only Shallow Extraction (No VRAM Teacher)", value=False, info="Extract ONLY the initial forces (Embeddings). Uses 0 VRAM for Teacher, lightning fast, creates a rough draft of the concept topography.")
                    knn_k_val = gr.Slider(minimum=2, maximum=150, step=1, value=8, label="KNN Neighbors (K) - ネットワークの太さ (KNNのみ)")
                    r_val = gr.Slider(minimum=8, maximum=256, step=8, value=64, label="LoRA Rank (r)")
                    ep_val = gr.Slider(minimum=1, maximum=100, step=1, value=40, label="Epochs")
                    start_btn = gr.Button("🚀 Start Distillation", variant="primary")
                with gr.Column():
                    output_log = gr.Textbox(label="Process Log", lines=20, max_lines=20, interactive=False)
                    
            start_btn.click(distill_process, inputs=[t_id, s_id, save_d, r_val, ep_val, knn_k_val, custom_concepts_text, algo_radio, low_vram_chk, shallow_chk, cache_dir_input], outputs=[output_log])
            
        # --- TAB 2: チャット & テスト ---
        with gr.TabItem("💬 Chat & Benchmark"):
            with gr.Row():
                base_txt = gr.Dropdown(choices=STUDENT_CHOICES, value="Qwen/Qwen2.5-0.5B-Instruct", label="Base Model", allow_custom_value=True, scale=3)
                
                with gr.Column(scale=3):
                    with gr.Row():
                        lora_dropdown = gr.Dropdown(choices=get_available_loras(), value="None", label="Select LoRA Topography", allow_custom_value=True, scale=4)
                        refresh_lora_btn = gr.Button("🔄", scale=1)
                
                use_lora_chk = gr.Checkbox(label="Apply LoRA", value=True)
                load_btn = gr.Button("Load Model", variant="secondary")
            
            refresh_lora_btn.click(refresh_lora_list, inputs=[], outputs=[lora_dropdown])
            
            load_status = gr.Textbox(label="Status", interactive=False)
            load_btn.click(load_model_for_chat, inputs=[base_txt, lora_dropdown, use_lora_chk, cache_dir_input], outputs=[load_status])
            
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        deepthink_chk = gr.Checkbox(label="🧠 Deepthink Mode (Concept Routing)", value=False)
                        lightweight_chk = gr.Checkbox(label="🍃 Lightweight Mode (FAISS Decoding - Edge Optimized)", value=False)
                    gr.ChatInterface(fn=chat_fn, additional_inputs=[deepthink_chk, lightweight_chk])
                with gr.Column(scale=1):
                    bench_btn = gr.Button("Run Full Benchmark", variant="primary")
                    bench_out = gr.Textbox(label="Benchmark Results", lines=15)
                    bench_btn.click(run_benchmark_ui, inputs=[], outputs=[bench_out])

        # --- TAB 3: 浅層抽出 ---
        with gr.TabItem("🔬 VRAM-Zero Shallow Extractor"):
            gr.Markdown("### Great-Circle Interference (全大円干渉型 概念核抽出)")
            gr.Markdown("テキストを入力すると、VRAMを一切消費せずにCPUのみで多重概念核（最も重要なキーワード群）を抽出します。")
            with gr.Row():
                with gr.Column(scale=3):
                    shallow_input = gr.Textbox(label="Input Text", value="We trained a neural network in Python to predict CRISPR off-target cuts in the genome.", lines=5)
                    shallow_btn = gr.Button("Extract Concepts", variant="primary")
                with gr.Column(scale=4):
                    shallow_output = gr.HTML(label="Concept Cores")
                    
            shallow_btn.click(extract_shallow_concepts, inputs=[shallow_input], outputs=[shallow_output])

if __name__ == "__main__":
    # ローカルでブラウザを自動で開く
    demo.launch(inbrowser=True)
