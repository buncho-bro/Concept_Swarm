import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from test_zero_shot_generality_stress import (
    HELD_OUT_TEST_CONCEPTS,
    BACKGROUND_CALIBRATION_CONCEPTS,
    load_six_real_images,
    STUDENT_MODEL_ID,
    LORA_PATH
)
from diagnose_topological_triangulation import extract_qwen_semantic_vec


def build_disjoint_512_calibration_lexicon() -> list[str]:
    """
    CLIPの512次元空間をフルランク(Rank=512, Nullspace=0)で張るための普遍背景辞書。
    【重要ルール】テスト対象の6概念（犬・車・花・寺院仏塔・海軍将校・ドット絵少女）に
    直接該当する単語は一切含めない（100% Hold-Out / Zero Leakage）。
    """
    # 94個の既存背景概念に加え、自然・物理・職業・日用品・芸術・科学・感情・空間属性など多様なアンカーを生成
    base_bg = [c[1] for c in BACKGROUND_CALIBRATION_CONCEPTS]
    extra_domains = [
        # Animals & Nature (NO dog/canine/puppy)
        "tiger stripes jungle predator", "bear grizzly forest hibernation", "deer antlers woodland grazing",
        "monkey primate climbing tree branch", "penguin antarctica ice swimming bird", "dolphin marine mammal jumping ocean",
        "whale giant ocean cetacean deep sea", "owl nocturnal bird feathers night", "snake reptile scales slithering",
        "crocodile swamp reptile jaws river", "bee insect honey hive wings", "spider web arachnid आठ legs",
        "crab crustacean beach shell claws", "octopus tentacles marine underwater", "camel desert humps sand caravan",
        "giraffe long neck savanna spots", "zebra stripes safari grassland", "kangaroo marsupial jumping pouch",
        "panda bamboo bear black and white", "koala eucalyptus tree marsupial", "fox red fur forest clever tail",
        "squirrel rodent acorn tree branch", "rabbit bunny ears hopping carrot", "hamster small rodent wheel cage",
        "peacock colorful tail feathers bird", "swan white bird lake graceful", "flamingo pink bird standing water",
        # Vehicles & Machines (NO car/automobile/sports car/suv)
        "tractor farm agriculture plowing field", "bulldozer construction earthmover blade", "crane tall construction lifting hook",
        "forklift warehouse pallets lifting", "sailboat wind canvas sails yacht", "canoe paddle river wooden boat",
        "kayak rapids water sport paddle", "ferry passenger boat harbor crossing", "jet ski water scooter waves",
        "hot air balloon colorful sky floating", "glider wings soaring wind aviation", "blimp airship zeppelin floating sky",
        "space shuttle orbit astronaut reentry", "satellite solar panels orbit space", "rover planetary exploration wheels mars",
        "scooter electric two wheels urban", "skateboard deck wheels street tricks", "roller skates wheels rink skating",
        "sled snow winter sliding downhill", "cable car gondola mountain ropeway", "ambulance emergency siren medical van",
        "fire_engine red ladder hose rescue", "garbage truck sanitation waste city", "tank armored military tracks cannon",
        # Plants & Landscapes (NO flower/blossom/dahlia/petals)
        "fern green fronds forest shade", "moss green rocks damp woodland", "seaweed kelp underwater ocean algae",
        "palm tree tropical beach coconut", "willow tree drooping branches river", "maple tree autumn red leaves",
        "vine climbing ivy wall green", "wheat golden grain harvest field", "corn stalks farm agriculture cob",
        "rice paddy water terraces green", "desert sand dunes arid sun", "glacier ice sheet blue frozen",
        "canyon steep rock cliff valley", "cave underground stalactites cavern", "island tropical surrounded by sea",
        "swamp marshland reeds murky water", "tundra frozen plain arctic moss", "prairie flat grassland horizon",
        "waterfall cascading water cliff mist", "geyser hot spring steam erupting", "coral reef underwater marine colorful",
        # Architecture & Places (NO pagoda/chinese temple)
        "windmill rotating blades rural grain", "igloo snow blocks dome arctic", "tent camping outdoor canvas shelter",
        "cabin wooden log forest retreat", "palace royal throne grand hall", "museum exhibition gallery artifacts",
        "library bookshelves reading quiet", "theater stage curtains auditorium seats", "cinema movie screen projector seats",
        "hospital medical clinic ward beds", "school classroom desks chalkboard", "university campus lecture hall",
        "Supermarket grocery aisles shopping cart", "bakery bread oven pastry shop", "cafe espresso table terrace",
        "hotel lobby reception accommodation", "airport terminal runway boarding gate", "train station platform railway clock",
        "harbor dock piers cargo cranes", "windmill turbine renewable energy", "observatory dome telescope stars",
        "greenhouse glass conservatory plants", "aquarium giant glass tank marine", "fountain water plaza stone sculpture",
        # People & Roles (NO naval officer/military uniform woman/admiral/pixel art/anime girl)
        "firefighter helmet hose rescue", "pilot aviation cockpit headset", "farmer agriculture hat field",
        "judge courtroom gavel legal robe", "musician playing instrument stage", "singer microphone concert vocal",
        "photographer holding camera viewfinder", "sculptor carving stone chisel", "writer typing manuscript desk",
        "librarian organizing books shelves", "architect blueprints building design", "engineer hard hat construction site",
        "mechanic wrench repairing engine", "carpenter sawing wood workshop", "plumber fixing pipes wrench",
        "electrician wiring circuit voltage", "baker kneading dough flour", "butcher cutting meat apron",
        "tailor sewing fabric measuring tape", "barber scissors haircut salon", "dentist dental chair teeth checkup",
        "nurse medical scrubs hospital care", "pharmacist medicine pills pharmacy", "veterinarian caring for cat clinic",
        "waiter serving tray restaurant", "receptionist front desk telephone", "cashier checkout register store",
        "janitor mop cleaning hallway", "lifeguard beach tower rescue float", "referee whistle sports field",
        "magician top hat wand stage illusion", "clown circus red nose juggling", "acrobat trapeze circus balance",
        "mountaineer climbing rope ice axe", "surfer riding ocean wave board", "skier snowy slope poles goggles",
        "swimmer pool lanes goggles diving", "cyclist racing helmet road bike", "boxer gloves ring punching bag",
        "archer bow arrow target aiming", "fencer foil mask duel sport", "golfer swinging club green course",
        # Visual Styles, Media & Objects (NO pixel art/anime/ドット絵)
        "watercolor painting wash paper", "charcoal drawing black smudge", "pastel chalk soft colors art",
        "origami folded paper crane craft", "pottery ceramic clay wheel kiln", "stained glass colorful window light",
        "mosaic tiles pattern ancient wall", "calligraphy ink brush lettering", "graffiti spray paint urban wall",
        "blueprint technical architectural diagram", "map cartography compass geography", "globe world sphere latitude",
        "microscope lens laboratory slide", "binoculars optical zoom birdwatching", "magnifying glass detective lens",
        "hourglass sand timer vintage", "compass magnetic needle north", "thermometer temperature mercury scale",
        "barometer atmospheric pressure weather", "scale balance weighing justice", "lantern glowing light camping night",
        "flashlight beam dark battery", "candle wax flame melted wick", "chandelier crystal hanging ceiling light",
        "mirror reflection glass frame", "vase ceramic container table", "basket woven wicker handle",
        "barrel wooden cask cellar storage", "crate wooden box shipping cargo", "suitcase luggage travel airport",
        "briefcase leather business office", "wallet leather cards coins pocket", "purse handbag strap fashion",
        "hat cap brim headwear sun", "scarf knitted wool winter neck", "gloves hand protection winter",
        "belt leather buckle waist", "tie necktie formal suit collar", "socks cotton footwear pair",
        "boots leather hiking rugged soles", "sandals summer beach open footwear", "helmet safety head protection",
        "crown tiara royal jewelled", "necklace chain pendant jewelry", "bracelet wrist ornament gold",
        "earrings jewelry lobe sparkle", "brooch pin decorative clasp", "medal ribbon award champion gold",
        "trophy cup winner podium sports", "flag banner waving pole wind", "bell metal ringing chime sound",
        "whistle blowing sound referee", "megaphone loud speaker cone", "microphone recording studio audio",
        "headphones ear cups listening music", "speaker audio sound woofer", "radio antenna broadcast tuner",
        "television screen living room broadcast", "projector lens beam presentation", "printer office paper ink",
        "scanner document digital copy", "calculator buttons math numbers", "typewriter vintage mechanical keys",
        "sewing machine needle thread fabric", "vacuum cleaner carpet household", "washing machine laundry drum",
        "refrigerator cold food kitchen", "microwave oven heating kitchen", "toaster bread slices breakfast",
        "blender smoothie kitchen blades", "kettle boiling water steam tea", "coffee maker brewing pot",
        "iron pressing clothes steam", "hair dryer blowing warm air", "fan rotating blades cooling air",
        "heater radiator warm room winter", "air conditioner cooling vent wall", "clock wall hands ticking time",
        "alarm clock bedside ringing morning", "stopwatch timing sport seconds", "calendar dates months schedule",
        "notebook spiral lined pages writing", "envelope letter stamp mail postal", "newspaper printed columns headlines",
        "magazine glossy cover articles", "dictionary reference definitions words", "encyclopedia volumes knowledge",
        "scroll parchment ancient rolled", "clipboard checklist paper clamp", "stapler office binding paper",
        "scissors cutting blades paper", "tape adhesive roll dispenser", "glue bottle sticking craft",
        "ruler measuring length markings", "protractor angles geometry math", "compass drawing circle pencil",
        "paintbrush bristles artist palette", "crayon wax colorful drawing child", "marker felt tip highlighter",
        "fountain pen ink nib writing", "pencil graphite eraser sketching", "chalk blackboard classroom dust",
        # Food, Materials, Science & Abstract
        "cheese dairy slice cheddar", "butter creamy spread dairy", "yogurt bowl spoon dairy",
        "egg shell yolk breakfast", "bacon crispy strips breakfast", "sausage grilled meat link",
        "steak grilled beef plate", "roast chicken dinner platter", "salmon fillet seafood lemon",
        "shrimp prawn seafood cooked", "lobster red shell claws seafood", "sushi roll rice seaweed fish",
        "noodles ramen bowl chopsticks", "spaghetti pasta tomato sauce fork", "taco tortilla filling mexican",
        "burrito wrapped tortilla beans", "sandwich bread slices lettuce", "Croissant flaky pastry french",
        "bagel round bread toasted", "pancake stack syrup breakfast", "waffle grid syrup breakfast",
        "donut glazed ring sweet", "muffin baked cup blueberry", "cookie chocolate chip baked",
        "pie crust slice fruit baked", "chocolate bar cocoa sweet", "candy colorful sweet wrapper",
        "popcorn bucket movie snack", "pretzel twisted baked salt", "chips crispy snack bowl",
        "nuts almonds walnuts bowl", "honey golden jar dipper", "jam fruit preserve jar",
        "soup warm bowl spoon broth", "salad green leaves vegetable bowl", "broccoli green vegetable florets",
        "carrot orange root vegetable", "potato tuber earth vegetable", "tomato red juicy vegetable",
        "onion layered bulb kitchen", "garlic cloves spice cooking", "pepper bell colorful vegetable",
        "cucumber green slices fresh", "corn cob yellow kernels", "peas green pod spherical",
        "pumpkin orange autumn squash", "watermelon green rind red slice", "pineapple tropical spiky fruit",
        "strawberry red berry seeds", "grapes bunch vine purple", "banana yellow curved fruit",
        "orange citrus peel segments", "lemon yellow sour citrus", "peach fuzzy sweet fruit",
        "cherry red pair stem", "pear green curved fruit", "mango tropical sweet fruit",
        "coconut brown shell white meat", "avocado green pit halves", "olive green branch oil",
        # Materials, Physics, Computing
        "crystal quartz mineral geometric", "gold bullion shiny metal bars", "silver coin metallic shiny",
        "copper wire conductive metal", "iron anvil heavy forged blacksmith", "steel beam structural girder",
        "marble stone veined polished", "granite rock speckled hard", "brick red clay masonry wall",
        "concrete gray cement pavement", "glass transparent fragile pane", "ceramic porcelain white glaze",
        "wood grain timber plank oak", "leather brown hide texture", "cotton soft white fiber",
        "silk smooth shiny fabric", "wool knitted yarn warm", "rubber elastic tire material",
        "plastic polymer molded synthetic", "paper white sheet cellulose", "cardboard corrugated box brown",
        "rope braided hemp knot", "chain metal links iron", "wire mesh fence metallic",
        "magnet horseshoe attraction poles", "battery electric power cell", "lightbulb glowing filament glass",
        "laser coherent light beam red", "prism rainbow spectrum refraction", "atom nucleus electrons physics",
        "molecule chemical bonds structure", "dna double helix genetics", "cell biology membrane nucleus",
        "fossil prehistoric imprint stone", "meteorite space rock crater", "comet icy tail solar system",
        "eclipse shadow moon sun", "aurora borealis northern lights sky", "rainbow colorful arc sky rain",
        "tornado funnel cloud wind vortex", "lightning thunderbolt electric storm", "snowflake hexagonal ice crystal",
        "icicle hanging frozen water", "dew drops morning leaf water", "fog mist low visibility",
        "Matrix linear algebra eigenvalues", "neural network deep learning weights", "quantum circuit qubit superposition",
        "cryptography cipher encryption key", "compiler syntax tree code", "operating system kernel process"
    ]
    return base_bg + extra_domains


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_concept_ids = [c[0] for c in HELD_OUT_TEST_CONCEPTS] + [c[0] for c in BACKGROUND_CALIBRATION_CONCEPTS]
    all_concept_texts = [c[1] for c in HELD_OUT_TEST_CONCEPTS] + [c[1] for c in BACKGROUND_CALIBRATION_CONCEPTS]
    N_unseen = len(HELD_OUT_TEST_CONCEPTS) # 6

    clip_model = SentenceTransformer("clip-ViT-B-32", device=device, local_files_only=True)
    m_minilm = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
    base_llm = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    llm = PeftModel.from_pretrained(base_llm, LORA_PATH) if os.path.exists(LORA_PATH) else base_llm
    llm.eval()

    real_images = load_six_real_images()
    img_objs = [item[2] for item in real_images]

    # Build expanded calibration lexicon (~370 concepts) — 0% overlap with the 6 Held-Out Test Concepts!
    calib_texts = build_disjoint_512_calibration_lexicon()
    print(f"Loaded {len(calib_texts)} Disjoint Calibration Concepts (0% overlap with 6 Held-Out Test Classes!)")

    # Encode all 100 evaluation valleys + all calibration concepts
    clip_eval_raw = F.normalize(torch.tensor(clip_model.encode(all_concept_texts), dtype=torch.float32, device=device), p=2, dim=-1) # (100, 512)
    clip_calib_raw = F.normalize(torch.tensor(clip_model.encode(calib_texts), dtype=torch.float32, device=device), p=2, dim=-1)      # (M, 512)
    clip_img_raw = F.normalize(torch.tensor(clip_model.encode(img_objs), dtype=torch.float32, device=device), p=2, dim=-1)           # (6, 512)

    mu_txt = clip_calib_raw.mean(dim=0, keepdim=True)
    mu_img = clip_img_raw.mean(dim=0, keepdim=True)

    clip_eval = F.normalize(clip_eval_raw - mu_txt, p=2, dim=-1)   # (100, 512)
    clip_calib = F.normalize(clip_calib_raw - mu_txt, p=2, dim=-1) # (M, 512)
    clip_img = F.normalize(clip_img_raw - mu_img, p=2, dim=-1)     # (6, 512)

    # Encode Idea Space for 100 evaluation valleys + calibration concepts
    qwen_eval = extract_qwen_semantic_vec(llm, tokenizer, all_concept_texts)
    qwen_calib = extract_qwen_semantic_vec(llm, tokenizer, calib_texts)

    minilm_eval_raw = F.normalize(torch.tensor(m_minilm.encode(all_concept_texts), dtype=torch.float32, device=device), p=2, dim=-1)
    minilm_calib_raw = F.normalize(torch.tensor(m_minilm.encode(calib_texts), dtype=torch.float32, device=device), p=2, dim=-1)
    mu_minilm = minilm_calib_raw.mean(dim=0, keepdim=True)
    minilm_eval = F.normalize(minilm_eval_raw - mu_minilm, p=2, dim=-1)
    minilm_calib = F.normalize(minilm_calib_raw - mu_minilm, p=2, dim=-1)

    torch.manual_seed(42)
    Q_orth, _ = torch.linalg.qr(torch.randn(2048, 384, device=device))
    idea_eval_raw = F.normalize(0.35 * qwen_eval + 0.65 * (minilm_eval @ Q_orth.T), p=2, dim=-1)
    idea_calib_raw = F.normalize(0.35 * qwen_calib + 0.65 * (minilm_calib @ Q_orth.T), p=2, dim=-1)

    mu_idea = idea_calib_raw.mean(dim=0, keepdim=True)
    idea_eval = F.normalize(idea_eval_raw - mu_idea, p=2, dim=-1)   # (100, 2048)
    idea_calib = F.normalize(idea_calib_raw - mu_idea, p=2, dim=-1) # (M, 2048)

    # Notice: The first 94 rows of calib_texts ARE the 94 background concepts (indices 6..99 of eval),
    # while calib_texts[94:] are completely disjoint extra background anchors!
    # Wait: What if we calibrate the bridge ONLY on calib_texts[94:] (the ~275 completely disjoint anchors
    # that are NOT in the 100 evaluation valleys at all!), so that ALL 100 evaluation valleys (both 0..5 AND 6..99)
    # are 100% Out-of-Sample?!
    print("\n" + "=" * 85)
    print(" 🏆 [Experiment A] 100% DISJOINT CALIBRATION BASIS (None of the 100 Valleys in Bridge!)")
    print("    Both the 6 Test Classes AND the 94 Distractor Classes are 100% Held-Out from Bridge!")
    print("=" * 85)
    B_clip_disjoint = clip_calib[94:] # (~275, 512) completely disjoint from all 100 eval concepts!
    B_idea_disjoint = idea_calib[94:] # (~275, 2048) completely disjoint from all 100 eval concepts!

    U_d, S_d, Vh_d = torch.linalg.svd(B_idea_disjoint, full_matrices=False)
    for r in [64, 96, 128, 192, 256]:
        r_eff = min(r, Vh_d.shape[0])
        V_sub = Vh_d[:r_eff].T # (2048, r_eff)
        # Project all 100 evaluation valleys onto V_sub — ALL 100 are now 100% Out-of-Sample!
        Y_eval_sub = F.normalize(idea_eval @ V_sub, p=2, dim=-1) # (100, r_eff)
        B_target_sub = F.normalize(B_idea_disjoint @ V_sub, p=2, dim=-1)

        for reg_c in [0.15, 0.30, 0.50]:
            W_dis = torch.linalg.solve(
                B_clip_disjoint.T @ B_clip_disjoint + reg_c * torch.eye(512, device=device),
                B_clip_disjoint.T @ B_target_sub
            )
            Z_img = F.normalize(clip_img @ W_dis, p=2, dim=-1)
            Z_bg  = F.normalize(B_clip_disjoint @ W_dis, p=2, dim=-1)
            hub_bias = torch.matmul(Y_eval_sub, Z_bg.T).mean(dim=-1)

            ranks = []
            top1, top3, top5 = 0, 0, 0
            for idx_img in range(N_unseen):
                sims = torch.matmul(Y_eval_sub, Z_img[idx_img]) - 0.5 * hub_bias
                rk = (torch.argsort(sims, descending=True) == idx_img).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rk)
                if rk == 1: top1 += 1
                if rk <= 3: top3 += 1
                if rk <= 5: top5 += 1
            print(f"  [Disjoint M={B_clip_disjoint.shape[0]}, r={r_eff:3d}, reg={reg_c:.2f}] Ranks: {ranks} | Top-1: {top1}/6 ({top1/6*100:.1f}%) | Top-3: {top3}/6 ({top3/6*100:.1f}%) | Top-5: {top5}/6 ({top5/6*100:.1f}%)")

    print("\n" + "=" * 85)
    print(" 🏆 [Experiment B] FULL 370-ANCHOR CALIBRATION + LOO ON THE 94 DISTRACTORS (6 Test Classes 100% Unseen)")
    print("=" * 85)
    for lam_loo in [0.20, 0.50]:
        G_all = idea_calib @ idea_calib.T
        M_all = idea_calib.shape[0]
        K_inv = torch.linalg.inv(G_all + lam_loo * torch.eye(M_all, device=device))
        H_all = G_all @ K_inv
        diag_H = torch.diag(H_all).unsqueeze(-1)
        idea_calib_loo = F.normalize((H_all @ idea_calib - diag_H * idea_calib) / (1.0 - diag_H), p=2, dim=-1)

        # Unseen 6 test classes projected onto the 370 calibration manifold
        G_unseen = idea_eval[:N_unseen] @ idea_calib.T
        Y_unseen = F.normalize((G_unseen @ K_inv) @ idea_calib, p=2, dim=-1)
        # Seen 94 distractor classes are the first 94 rows of idea_calib_loo!
        Y_eval_fair = torch.cat([Y_unseen, idea_calib_loo[:94]], dim=0) # (100, 2048)

        for reg_c in [0.15, 0.30, 0.50]:
            W_full = torch.linalg.solve(
                clip_calib.T @ clip_calib + reg_c * torch.eye(512, device=device),
                clip_calib.T @ idea_calib
            )
            Z_img = F.normalize(clip_img @ W_full, p=2, dim=-1)
            Z_bg  = F.normalize(clip_calib @ W_full, p=2, dim=-1)
            hub_bias = torch.matmul(Y_eval_fair, Z_bg.T).mean(dim=-1)

            ranks = []
            top1, top3, top5 = 0, 0, 0
            for idx_img in range(N_unseen):
                sims = torch.matmul(Y_eval_fair, Z_img[idx_img]) - 0.5 * hub_bias
                rk = (torch.argsort(sims, descending=True) == idx_img).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rk)
                if rk == 1: top1 += 1
                if rk <= 3: top3 += 1
                if rk <= 5: top5 += 1
            print(f"  [Full-370 lam_loo={lam_loo:.2f}, reg={reg_c:.2f}] Ranks: {ranks} | Top-1: {top1}/6 ({top1/6*100:.1f}%) | Top-3: {top3}/6 ({top3/6*100:.1f}%) | Top-5: {top5}/6 ({top5/6*100:.1f}%)")

if __name__ == "__main__":
    main()
