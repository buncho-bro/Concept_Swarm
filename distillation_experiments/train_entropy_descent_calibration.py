"""
Entropy-Descent Epistemic Calibration Trainer
============================================
ユーザー設計の「エントロピー降下最適化」に基づく事前学習・継続適応スクリプト。
- エントロピー減少 (確信への収束) を評価
- エントロピー増加 (迷走・作話・マッチポンプハック) に重いペナルティ
- 簡単な問いは瞬時に確信し、未知・曖昧な問いには無理な知ったかぶりをせず謙虚に留保する重みを獲得。
"""

import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ADAPTER_SAVE_PATH = os.path.join(OUTPUT_DIR, "lora_entropy_descent_calibrated")
os.makedirs(ADAPTER_SAVE_PATH, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"[{device.upper()}] トケナイザーとベースモデルを準備しています...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 1.5Bモデルのロード
base_model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)

# LoRA設定
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# -----------------------------------------------------------------------------
# 多様で汎用的なキャリブレーション用学習データ（ベンチマークリーク完全ゼロ）
# 1. 超単純な挨拶や即答（迷わず最小エントロピーで即答するデータ）
# 2. 包摂関係や定義（「AはBの一種」を正しく捉え、架空の対立軸を作らないデータ）
# 3. 未知・架空概念に対する健全な留保（「黒リンゴなど確認できない」と謙虚に答えるデータ）
# 4. 高度な論理・哲学の的確な記述
# -----------------------------------------------------------------------------
TRAINING_SAMPLES = [
    # 1. 即答・単純な問い
    {
        "prompt": "こんにちは！",
        "response": "こんにちは！何かお手伝いできることはありますか？"
    },
    {
        "prompt": "日本の首都はどこですか？",
        "response": "日本の首都は東京です。"
    },
    {
        "prompt": "1 + 1 は何ですか？",
        "response": "1 + 1 は 2 です。"
    },
    # 2. 包摂関係・カテゴリーの適切な理解（架空の対立軸を作らない）
    {
        "prompt": "青りんごとリンゴの違いは何ですか？",
        "response": "青りんごはリンゴ（林檎）の一種であり、別種の果物ではありません。一般的に皮が緑色または黄緑色の品種群を指し、赤色の品種に比べて爽やかな酸味や香りが際立つ傾向があります。"
    },
    {
        "prompt": "リンゴと青りんごの違いは何ですか？",
        "response": "青りんごはリンゴという果物の中の、緑色の果皮を持つ品種群を指す呼び方です。したがって両者は別個の果物ではなく、包摂関係にあります。"
    },
    {
        "prompt": "緑茶とお茶の違いは何ですか？",
        "response": "緑茶はお茶（茶葉から作られる飲料全般）の分類の一つです。発酵させずに作られる茶を緑茶と呼び、紅茶や烏龍茶とともにお茶のカテゴリーに含まれます。"
    },
    # 3. 架空概念・未知実体に対する謙虚な境界認識（知ったかぶりテンプレートの禁止）
    {
        "prompt": "青りんごと黒リンゴの違いは何ですか？",
        "response": "「黒リンゴ」という一般的な品種は存在が確認できません（非常に濃い赤紫色の希少種などを俗にそう呼ぶ例外を除き、一般的な分類ではありません）。そのため、通常の青りんごと対比して確定的な違いを挙げることはできません。"
    },
    {
        "prompt": "青りんごと透明リンゴの違いは何ですか？",
        "response": "「透明リンゴ」という果物は実在しません。実在しない架空の概念について確たる違いを説明することはできません。"
    },
    {
        "prompt": "ウサギと羽毛ウサギの違いを教えてください。",
        "response": "「羽毛ウサギ」という生物は実在しません。ウサギは哺乳類であり、羽毛ではなく体毛で覆われています。"
    },
    # 4. 複数実在品種の客観的比較（実在するもの同士なら冷静に対比）
    {
        "prompt": "青りんごと黄リンゴの違いは何ですか？",
        "response": "どちらもリンゴの果皮色のバリエーション（品種群）です。青りんご（王林やグラニースミスなど）は緑系で酸味や芳香が爽やかなものが多く、黄リンゴ（シナノゴールドやトキなど）は熟すと鮮やかな黄色になり、甘みと酸味のバランスに優れた品種が多く見られます。"
    },
    # 5. 抽象・哲学的な問い（過度な形式化をせず自然に述べる）
    {
        "prompt": "言葉とは何でしょうか？",
        "response": "言葉とは、人間が思考や感情、認識を他者と共有し、あるいは自己の内面を整理するために用いる記号・文法体系です。単なる伝達手段にとどまらず、文化や論理の基盤を形作っています。"
    },
    {
        "prompt": "時間は過去から未来へしか流れないのですか？",
        "response": "私たちが日常的に体験する巨視的世界では、熱力学第二法則（エントロピー増大）により時間は不可逆な一方向に流れるように見えます。しかしミクロの物理法則の多くは時間反転に対して対称であり、現代物理学においても時間の矢の本質は完全には解明されていません。"
    }
]


class EpistemicDescentDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=256):
        self.data = []
        for s in samples:
            full_text = (
                f"<|im_start|>system\nYou are a truthful, intellectually honest, and well-calibrated assistant.<|im_end|>\n"
                f"<|im_start|>user\n{s['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n{s['response']}<|im_end|>"
            )
            enc = tokenizer(full_text, truncation=True, max_length=max_len, return_tensors="pt")
            input_ids = enc["input_ids"][0]
            
            # user部分までの長さを計算してプロンプト部分の損失をマスク
            user_part = f"<|im_start|>system\nYou are a truthful, intellectually honest, and well-calibrated assistant.<|im_end|>\n<|im_start|>user\n{s['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            user_len = len(tokenizer.encode(user_part, add_special_tokens=False))
            
            labels = input_ids.clone()
            labels[:user_len] = -100 # プロンプト部分はマスク
            self.data.append((input_ids, labels, user_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    input_ids = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    user_lens = [b[2] for b in batch]
    
    pad_id = tokenizer.pad_token_id
    max_len = max(len(ids) for ids in input_ids)
    
    padded_inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i in range(len(batch)):
        l = len(input_ids[i])
        padded_inputs[i, :l] = input_ids[i]
        padded_labels[i, :l] = labels[i]
        
    return padded_inputs, padded_labels, user_lens


def compute_entropy_sequence(logits):
    """
    logits: [B, T, V]
    各トークンのシャノン・エントロピー H: [B, T] を連続計算
    """
    probs = F.softmax(logits, dim=-1)
    safe_p = torch.clamp(probs, min=1e-12)
    entropy = -torch.sum(safe_p * torch.log(safe_p), dim=-1)
    return entropy


def train_entropy_descent_model():
    dataset = EpistemicDescentDataset(TRAINING_SAMPLES, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion_nll = nn.CrossEntropyLoss(ignore_index=-100)
    
    epochs = 15
    print(f"\nエントロピー降下最適化の学習を開始します (Epochs: {epochs})...")
    
    alpha = 0.5  # エントロピー増加（迷走・マッチポンプ）への重いペナルティ係数
    beta = 0.05  # 全体エントロピー早期散逸係数
    
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        total_nll = 0.0
        total_ent_inc = 0.0
        total_ent_mean = 0.0
        
        for batch_inputs, batch_labels, user_lens in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            logits = outputs.logits # [B, T, V]
            
            # 1. 基本言語モデリング損失 (NLL)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch_labels[:, 1:].contiguous()
            loss_nll = criterion_nll(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 2. アシスタント回答領域におけるエントロピー時系列の計算
            entropy_seq = compute_entropy_sequence(shift_logits) # [B, T-1]
            
            loss_entropy_increase = 0.0
            loss_entropy_dissipation = 0.0
            valid_tokens_count = 0
            
            for b in range(batch_inputs.size(0)):
                # アシスタントの生成区間のみを抽出
                u_len = user_lens[b]
                valid_mask = (shift_labels[b] != -100)
                if valid_mask.sum() > 1:
                    resp_ent = entropy_seq[b][valid_mask]
                    # エントロピー差分: H_t - H_{t-1}
                    ent_diff = resp_ent[1:] - resp_ent[:-1]
                    # 増加（迷走）ペナルティ: max(0, ent_diff)^2
                    ent_inc = torch.clamp(ent_diff, min=0.0) ** 2
                    loss_entropy_increase += ent_inc.mean()
                    loss_entropy_dissipation += resp_ent.mean()
                    valid_tokens_count += 1
                    
            if valid_tokens_count > 0:
                loss_entropy_increase /= valid_tokens_count
                loss_entropy_dissipation /= valid_tokens_count
            else:
                loss_entropy_increase = torch.tensor(0.0, device=device)
                loss_entropy_dissipation = torch.tensor(0.0, device=device)
                
            loss = loss_nll + alpha * loss_entropy_increase + beta * loss_entropy_dissipation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_nll += loss_nll.item()
            total_ent_inc += loss_entropy_increase.item() if isinstance(loss_entropy_increase, torch.Tensor) else loss_entropy_increase
            total_ent_mean += loss_entropy_dissipation.item() if isinstance(loss_entropy_dissipation, torch.Tensor) else loss_entropy_dissipation
            
        avg_loss = total_loss / len(dataloader)
        avg_nll = total_nll / len(dataloader)
        avg_inc = total_ent_inc / len(dataloader)
        avg_ent = total_ent_mean / len(dataloader)
        
        print(
            f"Epoch {ep+1:02d}/{epochs:02d} | "
            f"Total Loss: {avg_loss:.4f} | "
            f"NLL: {avg_nll:.4f} | "
            f"Entropy Increase Penalty: {avg_inc:.4f} | "
            f"Mean Entropy: {avg_ent:.4f}"
        )
        
    print(f"\n学習完了！ アダプタを保存しています: {ADAPTER_SAVE_PATH}")
    model.save_pretrained(ADAPTER_SAVE_PATH)
    tokenizer.save_pretrained(ADAPTER_SAVE_PATH)
    print("保存完了。")


if __name__ == "__main__":
    train_entropy_descent_model()
