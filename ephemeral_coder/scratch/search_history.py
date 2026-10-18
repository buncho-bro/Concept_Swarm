import re

history_path = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\user_history.txt"

with open(history_path, "r", encoding="utf-8") as f:
    text = f.read()

entries = text.split("=== [Step ")
print(f"Total user steps: {len(entries)}")

for e in entries:
    if not e.strip():
        continue
    header = e.split("]")[0]
    content = e[len(header)+1:].strip()
    
    # Check if mentions 概念核 or 0VRAM or 蒸留 or 0VARM
    if any(k in content for k in ["概念核", "0VRAM", "0VARM", "蒸留"]):
        # print first few lines of this step
        lines = [l for l in content.splitlines() if l.strip()]
        first_few = " | ".join(lines[:3])[:150]
        print(f"Step {header}: {first_few}")
