import json
import re

history_path = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\user_history.txt"
out_path = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\filtered_history.txt"

with open(history_path, "r", encoding="utf-8") as f:
    text = f.read()

entries = text.split("=== [Step ")

keywords = ["概念核", "0VRAM", "0VARM", "蒸留", "Teacher", "Student", "Shallow", "全層固定", "KNN", "Chimera"]

with open(out_path, "w", encoding="utf-8") as out:
    for e in entries:
        if not e.strip():
            continue
        header = e.split("]")[0]
        content = e[len(header)+1:].strip()
        
        matches = [k for k in keywords if k.lower() in content.lower()]
        if matches:
            out.write(f"==================================================\n")
            out.write(f"=== STEP {header} (Matches: {', '.join(matches)}) ===\n")
            out.write(f"==================================================\n")
            out.write(content + "\n\n")

print("Done writing filtered_history.txt")
