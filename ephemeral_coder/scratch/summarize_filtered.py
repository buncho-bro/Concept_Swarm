import re

path = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\filtered_history.txt"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

sections = text.split("==================================================")
print(f"Total filtered sections: {len(sections)}")

for sec in sections:
    if "=== STEP " in sec:
        lines = sec.strip().splitlines()
        header = lines[0]
        body = "\n".join(lines[1:])
        # Let's inspect step number
        m = re.search(r"STEP (\d+)", header)
        if m:
            s_num = int(m.group(1))
            # print summary
            first_line = ""
            for l in lines[2:]:
                if l.strip() and not l.startswith("="):
                    first_line = l.strip()
                    break
            print(f"Step {s_num:4d}: {first_line[:90]}")
