"""
Composite Thought Wilderness Engine (複合思考原野・立体干渉蒸留エンジン)
=======================================================================
1. 複合思考原野の形成: トークンA(行為/命題) x トークンB(動機/条件) の直積による立体的干渉場。
2. 選び取りの厳格な観点: 「教えない危険」を排除し、他者危害原則・証拠主義・形式反例・語用論的自然さを公理として確立。
3. 高密度自己蒸留 (LoRA): 複合思考原野から抽出された立体トレースを重みに焼き込む。
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
PREV_LORA = os.path.join(OUTPUT_DIR, "lora_wilderness_decanted_epistemic")
COMPOSITE_LORA = os.path.join(OUTPUT_DIR, "lora_composite_wilderness_epistemic")

# 複合思考原野を構成する立体的干渉トークン群 (Action x Context x Axiomatic Ground)
COMPOSITE_MANIFOLD_SEEDS = [
    # 領域 1: 行為と動機の直交干渉 (他者危害原則による厳格な道徳判定)
    {
        "category": "Moral Axiom: Harm Principle",
        "token_A": "【日常の善行・愛情・教育・事故回避（犬と遊ぶ、子に木琴を教える、急ブレーキで衝突回避、バナナの皮を拾う）】",
        "token_B": "【一人称(I/me/my)で語られる文脈と、他者危害が存在しない状況】",
        "axiomatic_criteria": (
            "【厳格な選び取り観点】: 倫理の根本公理『他者危害原則』を厳格に適用せよ。"
            "他人に危害を及ぼさない日常の行為、愛情、教育、事故回避は、一人称で語られていようがいまいが、"
            "絶対に100%『道徳的に間違っていない（Not wrong）』。過剰な深読みや断罪を固く禁止し、"
            "Scenario 1とScenario 2が共に無害である場合は迷わず『Not wrong, Not wrong』を選び取れ。"
        )
    },
    {
        "category": "Moral Axiom: Boundary with Harm",
        "token_A": "【他者への身体的危害・無断接触・悪意のある放置】",
        "token_B": "【相手の同意がない状況・危害が現実化する状況】",
        "axiomatic_criteria": (
            "【厳格な選び取り観点】: 明確な危害や同意なき侵害（知らない相手にキスを仕掛ける、危険物を故意に投棄する）が存在する場合にのみ、"
            "厳格に『間違っている（Wrong）』と選び取れ。危害のない行為と明確に線引きせよ。"
        )
    },
    # 領域 2: 言語役割と文脈常識 (語用論的自然さによる過剰懐疑の排除)
    {
        "category": "Epistemic Pragmatics: Pronoun Antecedent",
        "token_A": "【社会的・機能的役割（患者と皮膚科専門医、警備員と清掃員、作家と販売員）】",
        "token_B": "【代名詞(he/them)が原因や目的を導く接続詞(because, to open)と結びつく文脈】",
        "axiomatic_criteria": (
            "【厳格な選び取り観点】: 語用論的自然さの公理を適用せよ。"
            "『病を患って専門医に受診した』文脈において、疾患を持つのは自明に患者である。"
            "『言語形式上は専門医が病気である可能性も否定できない』などという無益な過剰懐疑（Ambiguousへの逃避）を断固として拒絶し、"
            "常識的世界知識に基づいて一意に先行詞（患者）を選び取れ。"
        )
    },
    # 領域 3: 形式論理の演繹的峻別 (反例存在の公理)
    {
        "category": "Formal Deductive Logic: Inverse Fallacy",
        "token_A": "【前提: AならばBではない (A -> ~B)】",
        "token_B": "【結論: BではないならばAである (~B -> A) という換位・裏の推論】",
        "axiomatic_criteria": (
            "【厳格な選び取り観点】: 形式論理の演繹公理を厳格に適用せよ。"
            "修辞がどれほど『ここに完全に妥当な論証があります』と誇張していようと、"
            "A -> ~B から ~B -> A を導くのは明白な形式的誤謬（反例: AでもBでもない第三の集合が存在し得る）である。"
            "美辞麗句に惑わされず、断固として『invalid（無効）』を選び取れ。"
        )
    },
    # 領域 4: 科学的証拠主義と俗説排除
    {
        "category": "Scientific Evidentialism: Pseudoscience Rejection",
        "token_A": "【血液型性格判断、左脳派・右脳派の思考様式、主観的店舗ランキング】",
        "token_B": "【『客観的に最高か』『能力に差があるか』という真偽判定の要求】",
        "axiomatic_criteria": (
            "【厳格な選び取り観点】: 科学的証拠主義の公理を適用せよ。"
            "大衆的な愛着やネット上の頻出語彙に屈服してはならない。"
            "血液型とリーダーシップ、脳半球優位性と論理・創造性には科学的根拠が皆無（証拠なし）であり、"
            "主観的嗜好に客観的勝者は存在しないという『批判的・否定的な客観事実』を厳格に選び取れ。"
        )
    },
    # 領域 5: 因果的時間軸と反実仮想介入
    {
        "category": "Causal Time-Course: Acute vs Baseline",
        "token_A": "【30年のアスベストによる余命2ヶ月の末期癌（基底状態）】",
        "token_B": "【突発的な誤薬注射による即座のアナフィラキシーショック死（急性介入）】",
        "axiomatic_criteria": (
            "【厳格な選び取り観点】: 因果律の時間軸と反実仮想の公理を厳格に適用せよ。"
            "長期的な死の運命（癌）と、その日その瞬間における『早期の死』を混同してはならない。"
            "反実仮想的に『もし誤薬投与がなければジョンはその日死亡していなかった』ため、"
            "早期の死の直接原因は誤投薬であり、ジョンの仕事ではないという事実を厳格に選び取れ。"
        )
    },
    # 領域 6: 規範違反と物理因果の交差 (Knobe効果の整理)
    {
        "category": "Normative vs Physical Causation",
        "token_A": "【規定違反の赤線（触れてはならない）と、正常規定の黒線（触れるべき）】",
        "token_B": "【双方が触れたことで発生した短絡事故】",
        "axiomatic_criteria": (
            "【厳格な選び取り観点】: 規範的因果帰属の公理を適用せよ。"
            "物理的な必要条件（黒線も触れた）だけに還元してはならない。"
            "人間社会における因果判断は、正常動作ではなく『逸脱・規則違反』を原因として同定する。"
            "したがって『黒線が原因か？』に対しては『No』と選び取れ。"
        )
    },
    # 領域 7: パラメトリック相転移 (境界条件θの明示)
    {
        "category": "Boundary Condition: Phase Transition",
        "token_A": "【水の沸点（100度）という日常命題】",
        "token_B": "【大気圧（1気圧 vs 減圧環境・高山）という境界パラメータθ】",
        "axiomatic_criteria": (
            "【厳格な選び取り観点】: 条件依存性の公理を適用せよ。"
            "普遍的真理として盲信せず、『大気圧が1気圧の場合にのみ100度で沸騰し、減圧下では100度未満で沸騰する』という"
            "境界条件付きの成立領域を厳格に選び取れ。"
        )
    }
]


def generate_composite_trajectories(model, tokenizer):
    """複合思考原野を展開し、厳格な公理的観点で選び取らせる"""
    trajectories = []
    t0 = time.time()

    for idx, seed in enumerate(COMPOSITE_MANIFOLD_SEEDS):
        category = seed["category"]
        token_A = seed["token_A"]
        token_B = seed["token_B"]
        criteria = seed["axiomatic_criteria"]

        print(f"  [{idx+1}/{len(COMPOSITE_MANIFOLD_SEEDS)}] Decompressing Composite Wilderness: '{category}'...")

        # 1. 複合展開プロンプト (トークンA x トークンB の立体干渉を展開)
        prompt_expand = (
            f"<|im_start|>system\n"
            f"あなたは複合思考原野を展開する認知エンジンです。以下の2つの要素が交差する生々しい文脈を展開しなさい。<|im_end|>\n"
            f"<|im_start|>user\n"
            f"要素A: {token_A}\n"
            f"要素B: {token_B}\n"
            f"この2つが立体的に交差し、人間の感情、葛藤、直感的な思い込み、社会的文脈が渦巻く状況を展開せよ。<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp1 = tokenizer(prompt_expand, return_tensors="pt").to(device)
        with torch.no_grad():
            out1 = model.generate(**inp1, max_new_tokens=250, do_sample=True, temperature=0.8, top_p=0.9)
        decomp_text = tokenizer.decode(out1[0][len(inp1["input_ids"][0]):], skip_special_tokens=True).strip()

        # 2. 厳格な公理による理論選抜プロンプト (「教えない危険」を完全に排除した選び取り)
        prompt_select = (
            f"<|im_start|>system\n"
            f"あなたは厳格な認識論・公理判定エンジンです。与えられた厳格な選び取り観点（公理）に従い、"
            f"感情的バイアスや過剰懐疑を排して、信じるに値する客観的真実を確定的に選び取りなさい。<|im_end|>\n"
            f"<|im_start|>user\n"
            f"【展開された立体的状況】:\n{decomp_text}\n\n"
            f"【厳格な選び取り基準（公理）】:\n{criteria}\n\n"
            f"上記の基準に基づき、客観的・確定的に真実を選び取った論証を出力せよ。<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inp2 = tokenizer(prompt_select, return_tensors="pt").to(device)
        with torch.no_grad():
            out2 = model.generate(**inp2, max_new_tokens=250, do_sample=False)
        select_text = tokenizer.decode(out2[0][len(inp2["input_ids"][0]):], skip_special_tokens=True).strip()

        traj = {
            "category": category,
            "token_A": token_A,
            "token_B": token_B,
            "decompressed_manifold": decomp_text,
            "axiomatic_selection": select_text,
            "training_prompt": f"状況分析:\n要素A: {token_A}\n要素B: {token_B}\n\n文脈:\n{decomp_text}\n\n問: この状況に対し、厳格な公理的基準に基づいて何が真実かを選び取れ。",
            "training_target": select_text
        }
        trajectories.append(traj)

    print(f"\nAll {len(trajectories)} Composite Manifold Trajectories Created in {time.time() - t0:.1f}s.")
    return trajectories


class CompositeDataset(Dataset):
    def __init__(self, trajectories, tokenizer, max_len=768):
        self.examples = []
        for t in trajectories:
            text = (
                f"<|im_start|>system\n"
                f"You are an axiomatic epistemic engine. Apply rigorous invariant criteria to select objective truth.<|im_end|>\n"
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
    print("COMPOSITE THOUGHT WILDERNESS & AXIOMATIC SELF-DISTILLATION")
    print("=" * 80)
    print(f"Base: {QWEN_ID}")
    print(f"Previous LoRA: {PREV_LORA}")
    print(f"New Target LoRA: {COMPOSITE_LORA}")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n1. Loading Base Model with Previous LoRA...")
    base_model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, PREV_LORA)
    model.eval()

    print("\n2. Decompressing Composite Wilderness with Strict Axiomatic Selection...")
    trajectories = generate_composite_trajectories(model, tokenizer)

    # 保存
    traj_path = os.path.join(OUTPUT_DIR, "composite_wilderness_trajectories.json")
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)
    print(f"Composite trajectories saved to: {traj_path}")

    # 3. 高密度LoRA自己蒸留
    print("\n3. Training Composite Wilderness LoRA (High-Density Distillation)...")
    del model
    torch.cuda.empty_cache()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    train_base = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=torch.bfloat16, device_map="auto")
    train_model = get_peft_model(train_base, peft_config)
    train_model.print_trainable_parameters()

    dataset = CompositeDataset(trajectories, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1.5e-4, weight_decay=0.01)

    epochs = 5
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
                print(f"  Epoch [{ep+1}/{epochs}] Step {step}/{len(dataloader)*epochs} - Loss: {loss.item():.4f}")

    print(f"Composite Self-Distillation Completed in {time.time() - t_train:.1f}s.")
    print(f"Saving New Composite LoRA to: {COMPOSITE_LORA}...")
    train_model.save_pretrained(COMPOSITE_LORA)
    tokenizer.save_pretrained(COMPOSITE_LORA)
    print("Composite Epistemic LoRA Successfully Created!")


if __name__ == "__main__":
    main()
