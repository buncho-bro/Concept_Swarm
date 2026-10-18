import re

with open(r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\digest_concept_core_to_0vram.txt", "r", encoding="utf-8") as f:
    text = f.read()

sections = text.split("############################################################\n")

for sec in sections:
    if "=== STEP " not in sec:
        continue
    lines = sec.strip().splitlines()
    header = lines[0]
    body = "\n".join(lines[2:]).strip()
    m = re.search(r"STEP (\d+)", header)
    if m:
        s_num = int(m.group(1))
        # print first 200 chars of body
        first_p = body.split("\n\n")[0].replace("\n", " ")[:150]
        print(f"Step {s_num:4d} | {first_p}")
