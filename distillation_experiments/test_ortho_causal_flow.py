"""
test_ortho_causal_flow.py
-------------------------
直交因果偏向（真の質的差異）による結末アトラクター導出テスト
"""
import sys
import torch
import torch.nn.functional as F

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

W_embed = model.get_input_embeddings().weight.data.to(torch.float32)
vocab = tokenizer.get_vocab()

purified_words = []
purified_ids = []
for raw_s, tid in vocab.items():
    decoded = tokenizer.decode([tid])
    s_strip = decoded.strip()
    if not s_strip.isalpha() or not s_strip.isascii():
        continue
    has_space = raw_s.startswith('Ġ') or decoded.startswith(' ')
    if not has_space or len(s_strip) < 3:
        continue
    clean_w = s_strip.lower()
    if clean_w not in purified_words:
        purified_words.append(clean_w)
        purified_ids.append(tid)

ids_tensor = torch.tensor(purified_ids, device=device, dtype=torch.long)
selected_embeds = W_embed[ids_tensor]
mean_vec = selected_embeds.mean(dim=0, keepdim=True)
centered_embeds = selected_embeds - mean_vec
concept_space = F.normalize(centered_embeds, p=2, dim=-1)

def extract_flightpath(word: str):
    inputs = tokenizer(f" {word}", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states
    vecs = [h[0, -1, :].to(torch.float32) for h in hs]
    deltas = [vecs[i+1] - vecs[i] for i in range(len(vecs)-1)]
    fp = torch.stack(deltas).sum(dim=0)
    return F.normalize(fp, p=2, dim=0)

pairs = [
    ("crime", "police"),
    ("fire", "wood"),
    ("cloud", "cold"),
    ("group", "commutative"),
    ("strait", "blockade")
]

for w1, w2 in pairs:
    f1 = extract_flightpath(w1)
    f2 = extract_flightpath(w2)

    # 1. 量的中心（平均前進力）
    f_mid = F.normalize((f1 + f2) / 2.0, p=2, dim=0)

    # 2. 質的差異（AとBの張る平面に対する厳密な直交因果成分）
    # Gram-Schmidt直交化
    # u1 = f1
    u1 = f1
    # u2 = f2 - (f2 · u1) u1
    u2 = F.normalize(f2 - torch.dot(f2, u1) * u1, p=2, dim=0)

    # 思考粒子群を流す
    # 粒子の初期速度は前進力 + 直交成分への強い励起
    # 各層での非線形相互作用による直交トルク
    particles = f_mid.unsqueeze(0).repeat(2000, 1)
    noise = torch.randn_like(particles) * 0.05
    particles = F.normalize(particles + noise, p=2, dim=-1)

    # 直交偏向ベクトル（A-B軸に直角な新しい次元への推進力）
    # 全層フライトパスの後半層（Layer 14-28）の差分
    inputs_pair = tokenizer(f" {w1} {w2}", return_tensors="pt").to(device)
    with torch.no_grad():
        out_pair = model(**inputs_pair, output_hidden_states=True)
        hs_p = out_pair.hidden_states
    vecs_p = [h[0, -1, :].to(torch.float32) for h in hs_p]
    causal_spark = F.normalize(vecs_p[-1] - vecs_p[14], p=2, dim=0)

    # causal_sparkから入力語A, Bの成分を直交消去
    ortho_spark = causal_spark - torch.dot(causal_spark, u1) * u1 - torch.dot(causal_spark, u2) * u2
    ortho_spark = F.normalize(ortho_spark, p=2, dim=0)

    # 粒子の到達地点
    target_vec = F.normalize(f_mid * 0.4 + ortho_spark * 0.8, p=2, dim=0)

    sims = torch.mv(concept_space, target_vec)
    topk = torch.topk(sims, k=30)

    pair_words = {w1.lower(), w2.lower()}
    retrieved = []
    for idx in topk.indices.tolist():
        w = purified_words[idx]
        if w in pair_words or any(pw in w or w in pw for pw in pair_words):
            continue
        if len(w) >= 3 and w not in retrieved:
            retrieved.append(w)
        if len(retrieved) >= 6:
            break

    print(f"\n⚡ [{w1}] + [{w2}] ➔ 導出された直交因果アトラクター:")
    print(f"   {retrieved}")
