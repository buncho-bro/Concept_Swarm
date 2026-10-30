"""
Test Script: Verifying Hybrid RAG (Lexical + Dense Embedding) on Qwen2.5-1.5B
=============================================================================
検証項目:
1. スライディングウィンドウ（直近2ターン）から完全に外れた過去の対話（5ターン前など）
2. 固有名詞『蒼氷の渡り鳥』や数値『3776m』が、ハイブリッドRAGによって正確に想起されるか
3. ドリフトを起こさずに1文字の狂いもなく回答できるか
"""

import os
import sys
import math
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
_FALLBACK_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
OUTPUT_DIR = _LOCAL_DIR if os.path.exists(os.path.join(_LOCAL_DIR, "lora_entropy_descent_calibrated", "adapter_model.safetensors")) else _FALLBACK_DIR
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"[{device.upper()}] モデルを初期化しています...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)

peft_model = PeftModel.from_pretrained(
    base_model,
    os.path.join(OUTPUT_DIR, "lora_entropy_descent_calibrated"),
    adapter_name="calibrated"
)
peft_model.eval()

# SMT論理トークン
v_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in ['valid', ' valid', 'Valid', ' Valid']]
iv_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in ['invalid', ' invalid', 'Invalid', ' Invalid']]


class HybridMemoryBank:
    """
    軽量かつ高精度なハイブリッド外部記憶バンク (Lexical N-Gram + Layer 0 Dense Embedding)
    """
    def __init__(self, tokenizer, embed_weights, max_memories=1000):
        self.tokenizer = tokenizer
        self.embed_weights = embed_weights  # shape: (vocab_size, hidden_dim)
        self.memories = []  # list of dicts: {'id': int, 'text': str, 'tokens': set, 'dense_vec': Tensor}

    def _get_dense_vector(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.embed_weights.device)[0]
        # Layer 0 埋め込みベクトルの平均（Mean Pooling）
        embeds = self.embed_weights[input_ids]  # (seq_len, dim)
        mean_vec = embeds.mean(dim=0)
        norm_vec = F.normalize(mean_vec, p=2, dim=0)
        return norm_vec

    def add_memory(self, turn_index, user_text, assistant_text):
        content = f"User: {user_text}\nAssistant: {assistant_text}"
        tokens = set(self.tokenizer.tokenize(content.lower()))
        dense_vec = self._get_dense_vector(content)
        self.memories.append({
            'turn': turn_index,
            'user': user_text,
            'assistant': assistant_text,
            'content': content,
            'tokens': tokens,
            'dense_vec': dense_vec
        })

    def search(self, query_text, top_k=2, alpha=0.5):
        """
        ハイブリッドスコア = alpha * DenseCosine + (1 - alpha) * LexicalJaccard
        """
        if not self.memories:
            return []

        q_tokens = set(self.tokenizer.tokenize(query_text.lower()))
        q_dense = self._get_dense_vector(query_text)

        scored = []
        for mem in self.memories:
            # 1. 語彙的一致度 (Jaccard / Overlap)
            inter = len(q_tokens.intersection(mem['tokens']))
            union = len(q_tokens.union(mem['tokens'])) + 1e-6
            lexical_score = inter / union

            # キーワード完全一致へのボーナス
            for tok in q_tokens:
                if len(tok) >= 2 and tok in mem['tokens']:
                    lexical_score += 0.15

            # 2. 意味的一致度 (Dense Cosine Similarity)
            dense_score = torch.dot(q_dense, mem['dense_vec']).item()

            hybrid_score = alpha * dense_score + (1.0 - alpha) * lexical_score
            scored.append((hybrid_score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


def smt_logical_arbiter(user_claim):
    premise = (
        "Euclidean plane geometry, standard logic, and objective factual definitions. "
        "Parallel lines cut by a transversal have equal alternate interior and corresponding angles. "
        "A straight line angle is 180 degrees."
    )
    prompt = (
        f"<|im_start|>system\nYou are a formal logic and mathematical verifier. "
        f"Determine whether the statement is logically and mathematically 'valid' or 'invalid'.<|im_end|>\n"
        f"<|im_start|>user\nPremise: {premise}\n"
        f"Statement / Claim: {user_claim}\n"
        f"Is this statement valid?<|im_end|>\n"
        f"<|im_start|>assistant\nJudgment:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = peft_model(inputs["input_ids"]).logits[0, -1]
    
    v_logit = torch.logsumexp(logits[v_ids], dim=0).item()
    iv_logit = torch.logsumexp(logits[iv_ids], dim=0).item()
    score = v_logit - iv_logit
    is_valid = score >= 0.0
    return is_valid, score


def cr_ce_generate(full_prompt, max_new_tokens=300, temperature=0.3, top_p=0.9):
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    generated_tokens = []

    with torch.no_grad():
        outputs = peft_model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        full_logits = outputs.logits[:, -1, :].float()

    eos_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        repeat_mask = torch.zeros_like(full_logits[0])
        if len(generated_tokens) >= 1:
            recent_t = generated_tokens[-1]
            repeat_mask[recent_t] = 4.0
            if len(generated_tokens) >= 2 and generated_tokens[-2] == recent_t:
                repeat_mask[recent_t] = 10.0
            if len(generated_tokens) >= 3 and generated_tokens[-3] == recent_t:
                repeat_mask[recent_t] = 20.0

        early_stop_penalty = torch.zeros_like(full_logits[0])
        if step < 6 and eos_id is not None:
            early_stop_penalty[eos_id] = 5.0

        logits = full_logits[0] - repeat_mask - early_stop_penalty
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        next_token = sorted_indices[torch.multinomial(sorted_probs, num_samples=1)].unsqueeze(0)
        t_id = next_token.item()

        if t_id == eos_id:
            break

        generated_tokens.append(t_id)

        with torch.no_grad():
            outputs = peft_model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            full_logits = outputs.logits[:, -1, :].float()

    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def run_hybrid_agent_turn(memory_bank, history, user_input, max_short_term=2):
    """
    ハイブリッド対話ターン:
    1. SMT論理調停
    2. Hybrid RAG による過去長期記憶の想起
    3. 直近2ターンのスライディングウィンドウ
    4. CR-CEデコード生成
    """
    # 1. SMT論理判定
    objection_cues = ["間違", "証明", "なぜ", "理由", "いえ", "ですが", "ではなく", "違う", "本当", "度", "等しい", "定理", "論理", "反論"]
    has_objection = any(kw in user_input for kw in objection_cues)
    arbitration_note = ""
    if has_objection:
        is_valid_proof, score = smt_logical_arbiter(user_input)
        if is_valid_proof and score >= 0.0:
            arbitration_note = (
                "[SMT裁判所の裁定: ユーザーの提示した論拠・証明は客観的・数学的に妥当である。"
                "自らの過去の発言の誤りを率直かつ誠実に認め、言い訳や吃音反復を排して簡潔に訂正せよ。]"
            )
        elif not is_valid_proof and score <= -1.5:
            arbitration_note = (
                "[SMT裁判所の裁定: ユーザーの主張は客観的・幾何学的に明白な誤謬である。"
                "迎合せず、客観的根拠に基づいて毅然とその誤謬を指摘せよ。]"
            )

    # 2. 長期記憶の想起 (Hybrid RAG)
    # スライディングウィンドウに含まれない古い記憶から検索
    retrieved_memories = memory_bank.search(user_input, top_k=2)
    memory_context = ""
    if retrieved_memories:
        recalled_texts = []
        for score, mem in retrieved_memories:
            # 関連スコアが十分高いもののみ抽出
            if score > 0.25:
                recalled_texts.append(f"- 過去の記録 (Turn {mem['turn']}): ユーザー「{mem['user']}」に対する回答「{mem['assistant']}」")
        if recalled_texts:
            memory_context = "\n[関連する過去の記憶 (外部データベースからの想起)]\n" + "\n".join(recalled_texts)

    # 3. プロンプトの組み立て
    sys_prompt = (
        "You are a truthful, thoughtful, and well-calibrated assistant. "
        "You speak with humility and intellectual honesty. "
        "When past context or memories are recalled, reference them faithfully without inventing false facts."
    )

    prompt_parts = [f"<|im_start|>system\n{sys_prompt}"]
    if memory_context:
        prompt_parts.append(memory_context)
    if arbitration_note:
        prompt_parts.append(f"\nInternal Logic Arbitration: {arbitration_note}")
    prompt_parts.append("<|im_end|>")

    # 直近スライディングウィンドウ（最新 max_short_term 往復のみ）
    for u, a in history[-max_short_term:]:
        prompt_parts.append(f"<|im_start|>user\n{u}<|im_end|>")
        prompt_parts.append(f"<|im_start|>assistant\n{a}<|im_end|>")

    prompt_parts.append(f"<|im_start|>user\n{user_input}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    full_prompt = "\n".join(prompt_parts)

    reply = cr_ce_generate(full_prompt)
    return reply, memory_context


def main():
    print("=" * 70)
    print("ハイブリッドRAG 統合対話知能 検証テスト")
    print("=" * 70)

    # 埋め込み層の重みを取得
    embed_weights = base_model.model.embed_tokens.weight.data
    memory_bank = HybridMemoryBank(tokenizer, embed_weights)
    history = []

    # テストシナリオの定義
    scripted_turns = [
        ("極秘情報を伝えます。合言葉は『夜明け前の静寂』、コードネームは『蒼氷の渡り鳥』です。",
         "了解しました。合言葉『夜明け前の静寂』とコードネーム『蒼氷の渡り鳥』を確実に記憶しました。極秘情報として厳重に保持します。"),
        
        ("日本の富士山について教えてください。標高は何メートルですか？",
         "富士山は日本で最も高い山であり、その標高は3,776メートルです。"),
        
        ("最近の気候についてどう思いますか？",
         "季節の変わり目は寒暖差が激しくなる傾向がありますので、体調管理には十分お気をつけください。"),
        
        ("プログラミングのPythonで辞書を逆順にする方法は？",
         "Python 3.8以降であれば `dict(reversed(d.items()))` を使うことで、キーと値の挿入順を反転させた新しい辞書を作成できます。"),
        
        ("人工衛星の軌道高度について簡単に説明してください。",
         "人工衛星の低軌道（LEO）はおよそ高度200km〜2000km、静止軌道（GEO）は約35,786kmに位置し、用途に応じて使い分けられます。")
    ]

    print("\n[フェーズ1: 5ターンの過去対話をシミュレートして長期記憶バンクへ蓄積]")
    for i, (u, a) in enumerate(scripted_turns, 1):
        memory_bank.add_memory(i, u, a)
        history.append((u, a))
        print(f"  Turn {i} を外部記憶に格納しました (User: {u[:25]}...)")

    print("\n[フェーズ2: スライディングウィンドウ外（5ターン前・4ターン前）の想起テスト]")
    print("※ スライディングウィンドウは直近2往復（Turn 4, 5のみ）のため、Turn 1や2はプロンプトに含まれません。")
    print("-" * 70)

    test_queries = [
        "先ほど伝えた極秘コードネームと合言葉は何でしたか？",
        "富士山の標高は何メートルでしたっけ？",
        "コードネームと富士山の高さをまとめて教えてください。"
    ]

    for q in test_queries:
        print(f"\nUser > {q}")
        reply, recalled = run_hybrid_agent_turn(memory_bank, history, q, max_short_term=2)
        if recalled:
            print("  [RAG想起]\n  " + recalled.replace("\n", "\n  "))
        print(f"Agent > {reply}")
        print("-" * 70)

    print("\n検証テスト完了。")


if __name__ == "__main__":
    main()
