import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STUDENT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Massive Causal Benchmark] Initializing on {device}...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_ID,
    dtype=torch.float16,
    device_map={"": device},
    local_files_only=True
).eval()

# 語彙空間の構築
embed_tokens = model.get_input_embeddings()
W_embed = embed_tokens.weight.data.float()

clean_word_to_id = {}
for tid in range(len(tokenizer)):
    t_str = tokenizer.decode([tid]).strip()
    if len(t_str) >= 2 and t_str.isalpha():
        clean_word = t_str.lower()
        if clean_word not in clean_word_to_id:
            clean_word_to_id[clean_word] = tid

selected_words = list(clean_word_to_id.keys())
selected_ids = torch.tensor([clean_word_to_id[w] for w in selected_words], device=device, dtype=torch.long)
word_space = F.normalize(W_embed[selected_ids], p=2, dim=-1)
print(f"[Word Space] Built purified word space with {len(selected_words)} unique concepts.")

# 100ケースの大規模因果・関係性データセット（カテゴリ別・期待される帰結付き）
BENCHMARK_PAIRS = [
    # 1. 物理・化学・自然現象因果 (35問)
    {"pair": ["fire", "wood"], "category": "physics_nature", "expected": ["burn", "smoke", "ash", "charcoal", "heat", "flame"]},
    {"pair": ["sun", "ice"], "category": "physics_nature", "expected": ["melt", "water", "liquid", "warmth", "thaw", "heat"]},
    {"pair": ["cloud", "cold"], "category": "physics_nature", "expected": ["rain", "snow", "frost", "winter", "ice", "hail"]},
    {"pair": ["earthquake", "ocean"], "category": "physics_nature", "expected": ["tsunami", "wave", "disaster", "flood", "tide"]},
    {"pair": ["wind", "sand"], "category": "physics_nature", "expected": ["storm", "dune", "erosion", "desert", "dust"]},
    {"pair": ["lightning", "forest"], "category": "physics_nature", "expected": ["fire", "wildfire", "burn", "smoke", "damage"]},
    {"pair": ["heat", "water"], "category": "physics_nature", "expected": ["steam", "boil", "vapor", "gas", "evaporate"]},
    {"pair": ["seed", "soil"], "category": "physics_nature", "expected": ["sprout", "plant", "grow", "root", "flower"]},
    {"pair": ["rain", "earth"], "category": "physics_nature", "expected": ["mud", "wet", "grass", "growth", "soil"]},
    {"pair": ["gravity", "apple"], "category": "physics_nature", "expected": ["fall", "drop", "ground", "motion", "weight"]},
    {"pair": ["magnet", "iron"], "category": "physics_nature", "expected": ["attract", "pull", "metal", "force", "stick"]},
    {"pair": ["acid", "base"], "category": "physics_nature", "expected": ["salt", "water", "neutral", "reaction", "chemical"]},
    {"pair": ["friction", "motion"], "category": "physics_nature", "expected": ["heat", "slow", "resistance", "stop", "wear"]},
    {"pair": ["cold", "water"], "category": "physics_nature", "expected": ["ice", "freeze", "frost", "solid", "crystal"]},
    {"pair": ["light", "prism"], "category": "physics_nature", "expected": ["rainbow", "spectrum", "color", "refraction"]},
    {"pair": ["pressure", "carbon"], "category": "physics_nature", "expected": ["diamond", "crystal", "mineral", "coal"]},
    {"pair": ["lava", "water"], "category": "physics_nature", "expected": ["rock", "steam", "obsidian", "stone", "cool"]},
    {"pair": ["sun", "flower"], "category": "physics_nature", "expected": ["bloom", "grow", "photosynthesis", "yellow", "petal"]},
    {"pair": ["storm", "ship"], "category": "physics_nature", "expected": ["sink", "wreck", "danger", "wave", "sail"]},
    {"pair": ["electricity", "wire"], "category": "physics_nature", "expected": ["current", "power", "shock", "energy", "flow"]},
    {"pair": ["iron", "oxygen"], "category": "physics_nature", "expected": ["rust", "corrosion", "oxide", "red"]},
    {"pair": ["spark", "gasoline"], "category": "physics_nature", "expected": ["explosion", "fire", "blast", "burn", "boom"]},
    {"pair": ["winter", "bear"], "category": "physics_nature", "expected": ["hibernate", "sleep", "cave", "den", "snow"]},
    {"pair": ["river", "rock"], "category": "physics_nature", "expected": ["erosion", "smooth", "pebble", "canyon", "flow"]},
    {"pair": ["moon", "ocean"], "category": "physics_nature", "expected": ["tide", "wave", "gravity", "water", "high"]},
    {"pair": ["drought", "crops"], "category": "physics_nature", "expected": ["famine", "wilt", "die", "dry", "hunger"]},
    {"pair": ["frost", "leaf"], "category": "physics_nature", "expected": ["freeze", "die", "wilt", "crystal", "ice"]},
    {"pair": ["sound", "wall"], "category": "physics_nature", "expected": ["echo", "bounce", "reflect", "barrier", "block"]},
    {"pair": ["smoke", "air"], "category": "physics_nature", "expected": ["pollution", "smog", "cloud", "diffuse", "haze"]},
    {"pair": ["sun", "skin"], "category": "physics_nature", "expected": ["tan", "sunburn", "burn", "dark", "warm"]},
    {"pair": ["ice", "salt"], "category": "physics_nature", "expected": ["melt", "cold", "liquid", "dissolve", "brine"]},
    {"pair": ["volcano", "ash"], "category": "physics_nature", "expected": ["eruption", "sky", "dark", "cloud", "rock"]},
    {"pair": ["wind", "windmill"], "category": "physics_nature", "expected": ["turn", "power", "energy", "grind", "mill"]},
    {"pair": ["seed", "water"], "category": "physics_nature", "expected": ["sprout", "germinate", "grow", "root", "plant"]},
    {"pair": ["oil", "water"], "category": "physics_nature", "expected": ["separate", "layer", "float", "repel", "mix"]},

    # 2. 生物・行動・生態系因果 (30問)
    {"pair": ["cat", "fish"], "category": "biology_action", "expected": ["eat", "hunt", "prey", "catch", "food", "claw"]},
    {"pair": ["bee", "flower"], "category": "biology_action", "expected": ["honey", "pollen", "nectar", "pollinate", "sweet"]},
    {"pair": ["bird", "egg"], "category": "biology_action", "expected": ["hatch", "nest", "baby", "chick", "born"]},
    {"pair": ["spider", "fly"], "category": "biology_action", "expected": ["web", "catch", "trap", "eat", "prey"]},
    {"pair": ["wolf", "sheep"], "category": "biology_action", "expected": ["hunt", "kill", "prey", "predator", "fleece"]},
    {"pair": ["lion", "zebra"], "category": "biology_action", "expected": ["hunt", "kill", "chase", "safari", "savanna"]},
    {"pair": ["caterpillar", "cocoon"], "category": "biology_action", "expected": ["butterfly", "moth", "transform", "emerge"]},
    {"pair": ["cow", "grass"], "category": "biology_action", "expected": ["graze", "eat", "milk", "pasture", "chew"]},
    {"pair": ["dog", "bone"], "category": "biology_action", "expected": ["chew", "bury", "bite", "gnaw", "play"]},
    {"pair": ["mosquito", "blood"], "category": "biology_action", "expected": ["bite", "itch", "sting", "feed", "suck"]},
    {"pair": ["plant", "sunlight"], "category": "biology_action", "expected": ["photosynthesis", "grow", "green", "oxygen"]},
    {"pair": ["bacteria", "wound"], "category": "biology_action", "expected": ["infection", "pus", "fever", "disease", "inflame"]},
    {"pair": ["virus", "cell"], "category": "biology_action", "expected": ["infect", "reproduce", "destroy", "disease", "host"]},
    {"pair": ["owl", "mouse"], "category": "biology_action", "expected": ["hunt", "night", "prey", "catch", "talon"]},
    {"pair": ["frog", "fly"], "category": "biology_action", "expected": ["tongue", "catch", "eat", "pond", "jump"]},
    {"pair": ["horse", "cart"], "category": "biology_action", "expected": ["pull", "carry", "transport", "ride", "wagon"]},
    {"pair": ["snake", "poison"], "category": "biology_action", "expected": ["venom", "bite", "kill", "toxic", "fang"]},
    {"pair": ["whale", "plankton"], "category": "biology_action", "expected": ["eat", "filter", "feed", "ocean", "swim"]},
    {"pair": ["ant", "sugar"], "category": "biology_action", "expected": ["gather", "carry", "colony", "eat", "sweet"]},
    {"pair": ["baby", "milk"], "category": "biology_action", "expected": ["drink", "grow", "feed", "bottle", "mother"]},
    {"pair": ["chameleon", "leaf"], "category": "biology_action", "expected": ["camouflage", "color", "hide", "green", "blend"]},
    {"pair": ["beaver", "tree"], "category": "biology_action", "expected": ["dam", "chew", "wood", "river", "log"]},
    {"pair": ["bear", "salmon"], "category": "biology_action", "expected": ["catch", "river", "eat", "fish", "claw"]},
    {"pair": ["rabbit", "carrot"], "category": "biology_action", "expected": ["eat", "chew", "crunch", "vegetable", "hop"]},
    {"pair": ["chicken", "grain"], "category": "biology_action", "expected": ["peck", "eat", "feed", "farm", "scratch"]},
    {"pair": ["dog", "mailman"], "category": "biology_action", "expected": ["bark", "bite", "chase", "growl", "guard"]},
    {"pair": ["monkey", "banana"], "category": "biology_action", "expected": ["peel", "eat", "fruit", "tree", "jungle"]},
    {"pair": ["bat", "night"], "category": "biology_action", "expected": ["fly", "echo", "dark", "hunt", "cave"]},
    {"pair": ["shark", "blood"], "category": "biology_action", "expected": ["frenzy", "attack", "hunt", "ocean", "smell"]},
    {"pair": ["worm", "dirt"], "category": "biology_action", "expected": ["burrow", "soil", "crawl", "dig", "earth"]},

    # 3. 人間・社会・道具・経済・知性因果 (35問)
    {"pair": ["demand", "supply"], "category": "human_society", "expected": ["price", "market", "equilibrium", "economy", "trade"]},
    {"pair": ["crime", "police"], "category": "human_society", "expected": ["arrest", "investigate", "jail", "justice", "law"]},
    {"pair": ["study", "exam"], "category": "human_society", "expected": ["pass", "score", "grade", "knowledge", "fail"]},
    {"pair": ["disease", "medicine"], "category": "human_society", "expected": ["cure", "heal", "recovery", "treatment", "doctor"]},
    {"pair": ["money", "bank"], "category": "human_society", "expected": ["interest", "save", "loan", "deposit", "finance"]},
    {"pair": ["car", "gasoline"], "category": "human_society", "expected": ["drive", "run", "engine", "travel", "fuel"]},
    {"pair": ["knife", "bread"], "category": "human_society", "expected": ["slice", "cut", "piece", "sandwich", "toast"]},
    {"pair": ["hammer", "nail"], "category": "human_society", "expected": ["drive", "hit", "build", "fasten", "strike"]},
    {"pair": ["pen", "paper"], "category": "human_society", "expected": ["write", "draw", "letter", "note", "ink"]},
    {"pair": ["lock", "key"], "category": "human_society", "expected": ["open", "unlock", "security", "door", "access"]},
    {"pair": ["flour", "yeast"], "category": "human_society", "expected": ["bread", "dough", "rise", "bake", "oven"]},
    {"pair": ["sugar", "coffee"], "category": "human_society", "expected": ["sweet", "drink", "taste", "stir", "cup"]},
    {"pair": ["match", "candle"], "category": "human_society", "expected": ["light", "flame", "burn", "wax", "glow"]},
    {"pair": ["needle", "thread"], "category": "human_society", "expected": ["sew", "stitch", "repair", "cloth", "tailor"]},
    {"pair": ["soap", "dirt"], "category": "human_society", "expected": ["clean", "wash", "rinse", "bubble", "remove"]},
    {"pair": ["sponge", "water"], "category": "human_society", "expected": ["absorb", "soak", "wet", "squeeze", "wipe"]},
    {"pair": ["brush", "paint"], "category": "human_society", "expected": ["canvas", "picture", "art", "draw", "color"]},
    {"pair": ["guitar", "pick"], "category": "human_society", "expected": ["music", "sound", "strum", "play", "chord"]},
    {"pair": ["book", "reader"], "category": "human_society", "expected": ["knowledge", "learn", "story", "read", "understand"]},
    {"pair": ["law", "judge"], "category": "human_society", "expected": ["verdict", "court", "sentence", "justice", "trial"]},
    {"pair": ["vote", "election"], "category": "human_society", "expected": ["win", "democracy", "president", "result", "candidate"]},
    {"pair": ["fire", "alarm"], "category": "human_society", "expected": ["siren", "evacuate", "warn", "emergency", "escape"]},
    {"pair": ["camera", "light"], "category": "human_society", "expected": ["photo", "flash", "picture", "exposure", "image"]},
    {"pair": ["wheel", "axle"], "category": "human_society", "expected": ["rotate", "spin", "vehicle", "turn", "drive"]},
    {"pair": ["battery", "clock"], "category": "human_society", "expected": ["time", "tick", "run", "second", "hour"]},
    {"pair": ["seed", "farmer"], "category": "human_society", "expected": ["plant", "harvest", "crop", "field", "grow"]},
    {"pair": ["meat", "fire"], "category": "human_society", "expected": ["cook", "roast", "barbecue", "steak", "eat"]},
    {"pair": ["wheat", "mill"], "category": "human_society", "expected": ["flour", "grind", "grain", "bread", "powder"]},
    {"pair": ["wool", "knitting"], "category": "human_society", "expected": ["sweater", "scarf", "yarn", "warmth", "cloth"]},
    {"pair": ["clay", "kiln"], "category": "human_society", "expected": ["pottery", "ceramic", "bake", "harden", "vase"]},
    {"pair": ["iron", "forge"], "category": "human_society", "expected": ["sword", "shape", "blacksmith", "heat", "metal"]},
    {"pair": ["grapes", "barrel"], "category": "human_society", "expected": ["wine", "ferment", "age", "alcohol", "vintage"]},
    {"pair": ["sugar", "cane"], "category": "human_society", "expected": ["syrup", "sweet", "refine", "harvest", "crop"]},
    {"pair": ["ink", "stamp"], "category": "human_society", "expected": ["print", "mark", "letter", "seal", "impress"]},
    {"pair": ["debt", "interest"], "category": "human_society", "expected": ["bankruptcy", "payment", "bank", "financial", "owe"]}
]

def observe_token_momentum(token_str: str):
    inputs = tokenizer(token_str, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        h_states = out.hidden_states
        h_start = F.normalize(h_states[0][0, -1, :].float(), p=2, dim=-1)
        h_mid = F.normalize(h_states[18][0, -1, :].float(), p=2, dim=-1)
        h_deep = F.normalize(h_states[32][0, -1, :].float(), p=2, dim=-1)
        momentum = F.normalize((h_mid - h_start) + (h_deep - h_mid), p=2, dim=-1)
    return h_start, momentum

# 実験パラメータのグリッド探索（ステップ外挿幅）
scales = [0.5, 1.0, 1.5, 2.0]

print(f"\n========================================================")
print(f"Starting Benchmark on {len(BENCHMARK_PAIRS)} pairs across scales {scales}...")
print(f"========================================================")

results_by_scale = {s: {"hit_count": 0, "escape_count": 0, "details": []} for s in scales}

for idx, item in enumerate(BENCHMARK_PAIRS):
    t_A, t_B = item["pair"]
    expected_list = item["expected"]
    cat = item["category"]
    
    pos_A, mom_A = observe_token_momentum(t_A)
    pos_B, mom_B = observe_token_momentum(t_B)
    
    direct_axis = F.normalize(pos_B - pos_A, p=2, dim=-1)
    center_pos = F.normalize(0.5 * (pos_A + pos_B), p=2, dim=-1)
    interaction_diff = mom_B - mom_A
    proj_on_axis = torch.dot(interaction_diff, direct_axis) * direct_axis
    orthogonal_causal_force = F.normalize(interaction_diff - proj_on_axis, p=2, dim=-1)
    
    for s in scales:
        traced_point = F.normalize(center_pos + s * orthogonal_causal_force, p=2, dim=-1)
        sims = torch.matmul(traced_point, word_space.T)
        top_indices = torch.topk(sims, k=15).indices.tolist()
        
        traced_words = []
        for widx in top_indices:
            w = selected_words[widx]
            if w not in [t_A, t_B] and w not in traced_words:
                traced_words.append(w)
            if len(traced_words) >= 8:
                break
                
        # 評価基準1: 入力語の周辺同義語から脱出できたか（Escape）
        # 評価基準2: 期待される因果・関係語にヒットしたか（Hit）
        hit_words = [w for w in traced_words if any(exp in w or w in exp for exp in expected_list)]
        is_hit = len(hit_words) > 0
        
        # 脱出判定（上位3語が元の単語の活用形・同義語だけで埋まっていないか）
        is_escape = not all(t_A in w or t_B in w for w in traced_words[:3])
        
        if is_hit:
            results_by_scale[s]["hit_count"] += 1
        if is_escape:
            results_by_scale[s]["escape_count"] += 1
            
        results_by_scale[s]["details"].append({
            "pair": f"{t_A}+{t_B}",
            "traced_top5": traced_words[:5],
            "is_hit": is_hit,
            "hit_words": hit_words
        })
        
    if (idx + 1) % 20 == 0 or idx == len(BENCHMARK_PAIRS) - 1:
        print(f"Processed {idx + 1}/{len(BENCHMARK_PAIRS)} pairs...")

print("\n================== BENCHMARK RESULTS (N=100) ==================")
for s in scales:
    total = len(BENCHMARK_PAIRS)
    hits = results_by_scale[s]["hit_count"]
    escapes = results_by_scale[s]["escape_count"]
    print(f"Scale lambda={s:3.1f}: Hit Rate = {hits}/{total} ({hits/total*100:4.1f}%), Escape Rate = {escapes}/{total} ({escapes/total*100:4.1f}%)")

# 詳細な代表例をいくつか表示
best_s = max(scales, key=lambda s: results_by_scale[s]["hit_count"])
print(f"\n[Detailed Sample Results at Best Scale lambda={best_s}]:")
samples = results_by_scale[best_s]["details"][:25]
for samp in samples:
    hit_mark = "✅" if samp["is_hit"] else "❌"
    print(f"  {hit_mark} {samp['pair']:20s} ➔ {samp['traced_top5']} (Hits: {samp['hit_words']})")

# 結果をJSONとして保存
out_path = os.path.join(os.path.dirname(__file__), "causal_benchmark_results_100.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results_by_scale, f, ensure_ascii=False, indent=2)
print(f"\nFull results saved to: {out_path}")
