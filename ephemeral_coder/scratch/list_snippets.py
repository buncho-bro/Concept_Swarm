with open(r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\filtered_history.txt", "r", encoding="utf-8") as f:
    text = f.read()

entries = text.split("==================================================")

for e in entries:
    if "=== STEP " not in e:
        continue
    lines = [l for l in e.strip().splitlines() if l.strip()]
    header = lines[0]
    # find first non-header, non-line-number text
    body_lines = []
    for l in lines[1:]:
        if l.startswith("=== ") or l.startswith("(Line "):
            continue
        body_lines.append(l)
    
    snippet = " ".join(body_lines[:3])[:100]
    print(f"{header[:35]} | {snippet}")
