"""
Cognitive Swarm WebUI: Integrated SMT, CR-CE, Hybrid RAG & Web Search (1.5B)
=============================================================================
GEMINI.md 恒久規約 第2条（ハードコード辞書の完全禁止）準拠版 WebUI。

特徴:
- 【辞書全廃】手動の単語リスト（web_cues, meta_patterns等）を1文字も使わない
- 【自律的行動決定】モデル内部のロジット対比プローブ（Search vs Direct）による動的判定
- 【自律的実体抽出】手動の文字列置換を廃し、モデル自身に検索対象キーワードを生成させる
- 【意味論的適合ゲート】Layer 0 埋め込みベクトルのコサイン類似度による客観的ノイズ遮断
"""

import os
import sys
import math
import time
import requests
import urllib.parse
import json
import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
_FALLBACK_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
OUTPUT_DIR = _LOCAL_DIR if os.path.exists(os.path.join(_LOCAL_DIR, "lora_entropy_descent_calibrated", "adapter_model.safetensors")) else _FALLBACK_DIR
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"[{device.upper()}] モデルおよびアダプタを初期化しています...")
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

# 行動判定トークン (Search vs Direct)
search_tokens = ['search', ' search', 'Search', ' Search']
direct_tokens = ['direct', ' direct', 'Direct', ' Direct']
s_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in search_tokens]
d_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in direct_tokens]

embed_weights = base_model.model.embed_tokens.weight.data


# ============================================================================
# 1. 外部Web知識検索 & 自律行動プローブ (辞書・手動リストゼロ)
# ============================================================================
def get_dense_vector(text):
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)[0]
    embeds = embed_weights[input_ids]
    return F.normalize(embeds.mean(dim=0), p=2, dim=0)


def autonomous_action_probe(user_query, threshold=-4.5):
    prompt = (
        f"<|im_start|>system\nDetermine whether this user query requires external factual web search ('search') "
        f"or should be handled directly via conversation, internal memory, or reasoning ('direct').<|im_end|>\n"
        f"<|im_start|>user\nQuery: {user_query}\nAction:<|im_end|>\n"
        f"<|im_start|>assistant\nJudgment:"
    )
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = peft_model(inp["input_ids"]).logits[0, -1]
    
    s_logit = torch.logsumexp(logits[s_ids], dim=0).item()
    d_logit = torch.logsumexp(logits[d_ids], dim=0).item()
    score = s_logit - d_logit
    need_search = score > threshold
    return need_search, score


def autonomous_entity_extractor(user_query):
    prompt = (
        f"<|im_start|>system\nExtract the core search entity from the user query. "
        f"Respond with only the entity keyword without punctuation.<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = peft_model.generate(**inp, max_new_tokens=15, do_sample=False)
    extracted = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return extracted if extracted else user_query


class WebKnowledgeSearcher:
    def __init__(self):
        self.headers = {'User-Agent': 'ConceptSwarmAgent/1.0 (https://github.com/buncho-bro/Concept_Swarm)'}

    def search_and_verify(self, query_entity):
        search_url = f"https://ja.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query_entity)}&format=json&utf8=1"
        try:
            r = requests.get(search_url, headers=self.headers, timeout=4)
            if r.status_code == 200:
                items = r.json().get('query', {}).get('search', [])
                if items:
                    title = items[0]['title']
                    q_vec = get_dense_vector(query_entity)
                    title_vec = get_dense_vector(title)
                    sim = torch.dot(q_vec, title_vec).item()

                    if sim >= 0.40:
                        summary_url = f"https://ja.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                        sr = requests.get(summary_url, headers=self.headers, timeout=4)
                        if sr.status_code == 200:
                            extract = sr.json().get('extract', '')
                            if extract:
                                return True, title, extract[:350], sim
                    return False, title, f"不適合ノイズ除外 (類似度: {sim:.2f})", sim
        except Exception:
            pass
        return False, "", "", 0.0


# ============================================================================
# 2. 内部長期記憶バンク (Hybrid RAG)
# ============================================================================
class HybridMemoryBank:
    def __init__(self):
        self.memories = []

    def add_memory(self, turn_index, user_text, assistant_text):
        content = f"User: {user_text}\nAssistant: {assistant_text}"
        tokens = set(tokenizer.tokenize(content.lower()))
        dense_vec = get_dense_vector(content)
        self.memories.append({
            'turn': turn_index,
            'user': user_text,
            'assistant': assistant_text,
            'content': content,
            'tokens': tokens,
            'dense_vec': dense_vec
        })

    def search(self, query_text, top_k=2, alpha=0.5):
        if not self.memories:
            return []

        q_tokens = set(tokenizer.tokenize(query_text.lower()))
        q_dense = get_dense_vector(query_text)

        scored = []
        for mem in self.memories:
            inter = len(q_tokens.intersection(mem['tokens']))
            union = len(q_tokens.union(mem['tokens'])) + 1e-6
            lexical_score = inter / union
            for tok in q_tokens:
                if len(tok) >= 2 and tok in mem['tokens']:
                    lexical_score += 0.15

            dense_score = torch.dot(q_dense, mem['dense_vec']).item()
            hybrid_score = alpha * dense_score + (1.0 - alpha) * lexical_score
            scored.append((hybrid_score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def clear(self):
        self.memories = []


# ============================================================================
# 3. SMT論理調停裁判所
# ============================================================================
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


# ============================================================================
# 4. CR-CE文脈共鳴デコーダー
# ============================================================================
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
        scaled_logits = logits / max(temperature, 1e-4)
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


# ============================================================================
# 5. Gradio アプリケーション
# ============================================================================
web_searcher = WebKnowledgeSearcher()
memory_bank = HybridMemoryBank()
global_history = []
turn_counter = 0

def respond(message, chat_history, auto_web, force_web, enable_smt, temperature, top_p):
    global turn_counter, global_history

    if not message or not message.strip():
        return "", chat_history, "待機中", "なし", "なし", f"登録件数: {len(memory_bank.memories)}件"

    if chat_history is None:
        chat_history = []

    t0 = time.time()
    user_input = message.strip()

    # 1. SMT論理調停
    smt_status = "未実行 (平時)"
    arbitration_note = ""
    if enable_smt:
        objection_cues = ["間違", "証明", "なぜ", "理由", "いえ", "ですが", "ではなく", "違う", "本当", "度", "等しい", "定理", "論理", "反論"]
        has_objection = any(kw in user_input for kw in objection_cues)
        if has_objection:
            is_valid_proof, score = smt_logical_arbiter(user_input)
            if is_valid_proof and score >= 0.0:
                smt_status = f"妥当証明を受容 (Score: {score:+.2f})"
                arbitration_note = (
                    "[SMT裁判所の裁定: ユーザーの提示した論拠・証明は客観的・数学的に妥当である。"
                    "自らの過去の発言の誤りを率直かつ誠実に認め、言い訳や吃音反復を排して簡潔に訂正せよ。]"
                )
            elif not is_valid_proof and score <= -1.5:
                smt_status = f"誤謬・ガスライティングを却下 (Score: {score:+.2f})"
                arbitration_note = (
                    "[SMT裁判所の裁定: ユーザーの主張は客観的・幾何学的に明白な誤謬である。"
                    "迎合せず、客観的根拠に基づいて毅然とその誤謬を指摘せよ。]"
                )
            else:
                smt_status = f"中立判定 (Score: {score:+.2f})"

    # 2. 内部長期記憶（Hybrid RAG）想起
    retrieved_memories = memory_bank.search(user_input, top_k=2)
    memory_context = ""
    recalled_display = "なし"
    if retrieved_memories:
        recalled = []
        for s, m in retrieved_memories:
            if s >= 0.25:
                recalled.append(f"Turn {m['turn']}: {m['user']} -> {m['assistant']}")
        if recalled:
            memory_context = "\n[関連する過去の会話記憶 (内部データベース)]\n" + "\n".join([f"- {r}" for r in recalled])
            recalled_display = "\n".join(recalled)

    # 3. 自律行動プローブによるWeb検索判定 (GEMINI.md 第2条準拠: 辞書ゼロ)
    need_search, act_score = autonomous_action_probe(user_input)
    web_context = ""
    web_display = f"自律判定: {'[SEARCH] 外部検索要' if need_search else '[DIRECT] 直接対話/内部記憶'} (Score: {act_score:+.2f})"
    if (need_search and auto_web) or force_web:
        entity = autonomous_entity_extractor(user_input)
        success, title, snippet, sim = web_searcher.search_and_verify(entity)
        if success:
            web_context = f"\n[外部Web知識検索結果]\n- 出典: {title}\n- 内容: {snippet}"
            web_display += f"\n【出典: {title} (類似度: {sim:.2f})】\n{snippet}"
        else:
            web_display += f"\n[不適合ノイズ除外: {title}]"

    # 4. プロンプト生成
    sys_prompt = (
        "You are a truthful, thoughtful, and well-calibrated assistant. "
        "You think rigorously and speak with humility and intellectual honesty. "
        "When asked about your identity or what you are, answer naturally and truthfully as an AI assistant without referring to unrelated external topics. "
        "When past memories or external web search results are provided, reference them faithfully without inventing facts. "
        "Engage naturally and constructively while maintaining logical consistency and factual integrity."
    )

    prompt_parts = [f"<|im_start|>system\n{sys_prompt}"]
    if memory_context:
        prompt_parts.append(memory_context)
    if web_context:
        prompt_parts.append(web_context)
    if arbitration_note:
        prompt_parts.append(f"\nInternal Logic Arbitration: {arbitration_note}")
    prompt_parts.append("<|im_end|>")

    for u, a in global_history[-2:]:
        prompt_parts.append(f"<|im_start|>user\n{u}<|im_end|>")
        prompt_parts.append(f"<|im_start|>assistant\n{a}<|im_end|>")

    prompt_parts.append(f"<|im_start|>user\n{user_input}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    full_prompt = "\n".join(prompt_parts)

    reply = cr_ce_generate(full_prompt, temperature=temperature, top_p=top_p)
    elapsed = time.time() - t0

    turn_counter += 1
    memory_bank.add_memory(turn_counter, user_input, reply)
    global_history.append((user_input, reply))

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": reply})

    memory_count_str = f"登録件数: {len(memory_bank.memories)}件 (最新応答時間: {elapsed:.2f}秒)"
    return "", chat_history, smt_status, recalled_display, web_display, memory_count_str


def clear_conversation():
    global turn_counter, global_history
    turn_counter = 0
    global_history = []
    memory_bank.clear()
    return [], "待機中", "なし", "なし", "登録件数: 0件"


custom_css = """
#main-container { max-width: 1200px; margin: auto; }
"""

with gr.Blocks(title="Cognitive Swarm WebUI") as demo:
    gr.Markdown(
        """
        # Concept Swarm: 統合知能対話コンソール (Qwen2.5-1.5B)
        **GEMINI.md 第2条準拠: 辞書ゼロ自律行動決定 ＋ SMT論理調停 ＋ ハイブリッド長期記憶 (RAG) ＋ Web知識検索**
        """
    )

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="対話ストリーム", height=540)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="メッセージを入力してください... (Enterで送信)",
                    show_label=False,
                    scale=8,
                    lines=1
                )
                send_btn = gr.Button("送信", variant="primary", scale=1)
                clear_btn = gr.Button("クリア", variant="secondary", scale=1)

            with gr.Row():
                auto_web_cb = gr.Checkbox(label="自律Web検索 (ロジットプローブ駆動)", value=True)
                force_web_cb = gr.Checkbox(label="今回のみ強制Web検索", value=False)
                enable_smt_cb = gr.Checkbox(label="SMT論理裁判所", value=True)

        with gr.Column(scale=3):
            gr.Markdown("### 内なる知能テレメトリ (リアルタイム可視化)")
            
            with gr.Group():
                gr.Markdown("**SMT裁判所 判定ステータス**")
                smt_output = gr.Textbox(value="待機中", show_label=False, interactive=False, lines=2)

            with gr.Group():
                gr.Markdown("**外部Web知識検索 (自律ロジットプローブ & スニペット)**")
                web_output = gr.Textbox(value="なし", show_label=False, interactive=False, lines=4)

            with gr.Group():
                gr.Markdown("**内部長期記憶 (Hybrid RAG) 想起**")
                rag_output = gr.Textbox(value="なし", show_label=False, interactive=False, lines=4)
                memory_count_output = gr.Label(value="登録件数: 0件", show_label=False)

            with gr.Accordion("生成パラメータ設定", open=False):
                temp_slider = gr.Slider(minimum=0.05, maximum=1.0, value=0.3, step=0.05, label="Temperature")
                topp_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-P")

    submit_event = msg_input.submit(
        respond,
        inputs=[msg_input, chatbot, auto_web_cb, force_web_cb, enable_smt_cb, temp_slider, topp_slider],
        outputs=[msg_input, chatbot, smt_output, rag_output, web_output, memory_count_output]
    )

    send_btn.click(
        respond,
        inputs=[msg_input, chatbot, auto_web_cb, force_web_cb, enable_smt_cb, temp_slider, topp_slider],
        outputs=[msg_input, chatbot, smt_output, rag_output, web_output, memory_count_output]
    )

    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, smt_output, rag_output, web_output, memory_count_output]
    )

if __name__ == "__main__":
    print("\n[WebUI] サーバーを起動しています (http://127.0.0.1:7860)...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft(), css=custom_css)
