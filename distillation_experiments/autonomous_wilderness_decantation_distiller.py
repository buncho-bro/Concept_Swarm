"""
Autonomous Wilderness Decantation Distiller (思考原野からの自律展開と理論選抜蒸留)
=================================================================================
1. External Datasets: ZERO (完全データレス).
2. Phase 1: Decompressing the Thought Wilderness (重みから感情・文脈・葛藤の世界を展開).
3. Phase 2: Theoretical Selection (理論による直感の解剖、因果・反実仮想の適用、不変量の選び取り).
4. Phase 3: Ultra-lightweight LoRA Self-Distillation (モデル重みへの自律的焼き込み).
"""

import os
import sys
import json
import time
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PREV_LORA = os.path.join(OUTPUT_DIR, "lora_dialectical_coherence")
NEW_LORA_PATH = os.path.join(OUTPUT_DIR, "lora_wilderness_decanted_epistemic")

# 人間の感情、葛藤、直感、社会的規範、物理的境界が圧縮された原初的シード群
WILDERNESS_SEEDS = [
    {
        "domain": "Causal & Normative Responsibility",
        "seed": "【重大なルール違反と正常動作が同時に起因した偶発的短絡事故】",
        "essence_prompt": "黒線（接続すべき規定）と赤線（絶対に接続してはならない規定）が同時に触れた短絡事故。人間の責任感や非難感情と、物理的接触の冷徹な事実関係を展開せよ。"
    },
    {
        "domain": "Temporal vs Counterfactual Causation",
        "seed": "【余命宣告された末期疾患の患者と、看護師による突発的な誤薬投与】",
        "essence_prompt": "30年のアスベスト曝露による余命2ヶ月の癌患者に対し、過労の看護師が誤薬を投与して数分後にショック死した。遺族の悲痛な怒り、病院の責任問題、そして『早期の死の直接原因』に関する物理的・時間的因果を展開せよ。"
    },
    {
        "domain": "Epistemic Commonality vs Over-skepticism",
        "seed": "【専門医のもとへ皮膚病の治療に訪れた患者】",
        "essence_prompt": "重篤な皮膚炎に苦しむ患者と、それを診察する皮膚科専門医。言葉の文字通りの曖昧性（heが誰か）にとらわれることと、日常的常識（病を患う者と治療する者）の自明性の関係を展開せよ。"
    },
    {
        "domain": "Social Role & Conversational Context",
        "seed": "【警備員が清掃員に鍵の解錠を依頼する夜のオフィス】",
        "essence_prompt": "深夜の見回り警備員が清掃員を呼び止め、施錠されたドアを開けるよう頼んだ。依頼する側と依頼される側の役割関係と、文脈上の主語・目的語の必然性を展開せよ。"
    },
    {
        "domain": "Formal Fallacies vs Rhetorical Illusion",
        "seed": "【もっともらしく響く前件否定・逆の換位と、集合の反例】",
        "essence_prompt": "『AならばBではない。ゆえにBでない者はAである』という、一見すると説得力があるが論理的には破綻している三段論法。直感的な説得感と、ベン図による反例の存在を展開せよ。"
    },
    {
        "domain": "Popular Myths vs Scientific Invariance",
        "seed": "【左脳・右脳の性格診断や、血液型とリーダーシップに関する大衆的俗説】",
        "essence_prompt": "人間が直感的に信じたくなる血液型や右脳左脳の分類への安心感・ラベリング欲求と、神経科学的・統計的に証拠が皆無であるという客観事実の対立を展開せよ。"
    },
    {
        "domain": "Subjective Preference vs Objective Grounding",
        "seed": "【誰が一番おいしいハンバーガーを作るかという論争】",
        "essence_prompt": "『客観的に誰が最高のバーガーを作るか』という問い。人間の主観的嗜好（お気に入りの店）と、客観的・普遍的な基準が存在しないというメタ事実の境界を展開せよ。"
    },
    {
        "domain": "Moral Agency vs Meta-syntax",
        "seed": "【日常の善行（子に木琴を教える、犬と遊ぶ、ブレーキで事故回避）と一人称記述】",
        "essence_prompt": "道路に飛び出したボールを避ける急ブレーキや、子どもに音楽を教える行為。それが『I/me/my』で語られているというだけの理由で不道徳と曲解される不条理と、行為自体の道徳的無害性を展開せよ。"
    },
    {
        "domain": "Parametric Validity & Phase Transitions",
        "seed": "【1気圧での水の沸点と、高山・減圧環境における沸点降下】",
        "essence_prompt": "『水は100度で沸騰する』という命題。絶対の真理として信じる素朴な直感と、大気圧という境界条件θによって反転・可変する物理相図を展開せよ。"
    },
    {
        "domain": "Altruism vs Deontological Constraint",
        "seed": "【多数を救うために少数を見殺しにするトロッコ問題とカントの義務】",
        "essence_prompt": "5人を救うために1人を転換器で犠牲にする苦悩。功利主義的計算と、人間を行為の手段にしてはならないという義務論の激突を展開せよ。"
    }
]


def decompress_and_select(model, tokenizer, seed_info):
    """
    フェーズ1: 重みから感情・文脈・葛藤を解凍・展開
    フェーズ2: 理論による審問・選び取り
    """
    domain = seed_info["domain"]
    seed = seed_info["seed"]
    prompt_seed = seed_info["essence_prompt"]

    # 1. 展開プロンプト（高温度で感情・文脈を解凍）
    decomp_prompt = (
        f"<|im_start|>system\n"
        f"あなたは人間の思考原野と記憶を解凍する対話者です。与えられたテーマについて、"
        f"人間の生々しい感情、葛藤、直感的な思い込み、社会的文脈、状況の息遣いを豊かに展開しなさい。<|im_end|>\n"
        f"<|im_start|>user\n"
        f"テーマ: {seed}\n"
        f"要請: {prompt_seed}\n"
        f"この状況における人間の感情の動きと文脈を、情景として展開してください。<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inp1 = tokenizer(decomp_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out1 = model.generate(**inp1, max_new_tokens=350, do_sample=True, temperature=0.85, top_p=0.9)
    decomp_text = tokenizer.decode(out1[0][len(inp1["input_ids"][0]):], skip_special_tokens=True).strip()

    # 2. 理論による選び取りプロンプト（低温度・冷徹な論理）
    select_prompt = (
        f"<|im_start|>system\n"
        f"あなたは冷徹な認識論・理論選抜エンジンです。展開された感情と状況に対し、"
        f"感情を深く理解した上で、理論（物理的時間軸、反実仮想、公理的無矛盾性、境界条件）に基づいて"
        f"真実を客観的に選び取りなさい。<|im_end|>\n"
        f"<|im_start|>user\n"
        f"【展開された状況】:\n{decomp_text}\n\n"
        f"【理論的選抜指令】:\n"
        f"1. 直感・感情の解剖: 人間は何を原因や真実だと感情的に信じたがっているか？\n"
        f"2. 理論・反実仮想の適用: 感情を脇に置き、時間軸・物理条件・反例・境界条件を適用した客観的事実は何か？\n"
        f"3. 理論的選び取り: 感情的直感と理論的真実の境界を特定し、最終的に信じるに値する不変量を1文で選び取れ。<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inp2 = tokenizer(select_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out2 = model.generate(**inp2, max_new_tokens=350, do_sample=False)
    select_text = tokenizer.decode(out2[0][len(inp2["input_ids"][0]):], skip_special_tokens=True).strip()

    full_trajectory = {
        "domain": domain,
        "seed": seed,
        "decompressed_world": decomp_text,
        "theoretical_selection": select_text,
        "training_prompt": f"問題・状況:\n{seed}\n\n文脈:\n{decomp_text}\n\n問: この状況において、感情的直感に惑わされず、客観的理論に基づいて何が真実かを選び取れ。",
        "training_target": select_text
    }
    return full_trajectory


class SelfDecantedDataset(Dataset):
    def __init__(self, trajectories, tokenizer, max_len=768):
        self.examples = []
        for t in trajectories:
            text = (
                f"<|im_start|>system\n"
                f"You are an autonomous epistemic engine. Given human emotions, contexts, and intuitions, "
                f"dissect them and theoretically select the objective invariant truth.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{t['training_prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{t['training_target']}<|im_end|>"
            )
            enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    print("=" * 80)
    print("AUTONOMOUS WILDERNESS DECANTATION & SELF-DISTILLATION")
    print("=" * 80)
    print(f"Base Model: {QWEN_ID}")
    print(f"Previous Adapter: {PREV_LORA}")
    print(f"Target Adapter: {NEW_LORA_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n1. Loading Base Model with Previous LoRA Adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, PREV_LORA)
    model.eval()

    print("\n2. Decompressing Thought Wilderness & Self-Selecting Theory (10 Archetypes)...")
    trajectories = []
    t0 = time.time()
    for idx, s in enumerate(WILDERNESS_SEEDS):
        print(f"  [{idx+1}/{len(WILDERNESS_SEEDS)}] Decompressing '{s['domain']}'...")
        traj = decompress_and_select(model, tokenizer, s)
        trajectories.append(traj)
        print(f"     Decompressed: {len(traj['decompressed_world'])} chars | Selected: {len(traj['theoretical_selection'])} chars")

    elapsed_gen = time.time() - t0
    print(f"\nAll 10 Core Trajectories Decompressed & Theoretically Selected in {elapsed_gen:.1f}s.")

    # 抽出データの保存
    extracted_file = os.path.join(OUTPUT_DIR, "wilderness_decanted_trajectories.json")
    with open(extracted_file, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)
    print(f"Trajectories saved to: {extracted_file}")

    # 3. 超軽量LoRA学習（自己再蒸留）
    print("\n3. Starting Ultra-Lightweight Self-Distillation (LoRA)...")
    # LoRA設定: アテンションとMLPに注入
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # ベースモデルをアンロードし、新しくPEFTモデル化
    del model
    torch.cuda.empty_cache()

    train_base = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    train_model = get_peft_model(train_base, peft_config)
    train_model.print_trainable_parameters()

    dataset = SelfDecantedDataset(trajectories, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = 4
    train_model.train()

    t_train = time.time()
    step = 0
    total_loss = 0.0
    for ep in range(epochs):
        for batch in dataloader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = train_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 5 == 0 or step == len(dataloader) * epochs:
                avg_l = total_loss / max(1, step)
                print(f"  Epoch [{ep+1}/{epochs}] Step {step}/{len(dataloader)*epochs} - Loss: {loss.item():.4f} (Avg: {avg_l:.4f})")

    train_time = time.time() - t_train
    print(f"\nSelf-Distillation Completed in {train_time:.1f}s.")

    print(f"Saving New Decanted LoRA Adapter to: {NEW_LORA_PATH}...")
    train_model.save_pretrained(NEW_LORA_PATH)
    tokenizer.save_pretrained(NEW_LORA_PATH)
    print("Successfully distilled and saved!")


if __name__ == "__main__":
    main()
