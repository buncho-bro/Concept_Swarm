import json
import re

history_path = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\filtered_history.txt"
out_path = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\digest_concept_core_to_0vram.txt"

with open(history_path, "r", encoding="utf-8") as f:
    text = f.read()

entries = text.split("==================================================")

with open(out_path, "w", encoding="utf-8") as out:
    for e in entries:
        if "=== STEP " not in e:
            continue
        # header
        lines = e.strip().splitlines()
        header = lines[0]
        # find the actual content
        content_lines = []
        for line in lines[1:]:
            if line.startswith("(Line ") or line.startswith("==="):
                continue
            content_lines.append(line)
        content = "\n".join(content_lines).strip()
        
        out.write(f"\n{'#'*60}\n")
        out.write(f"{header}\n")
        out.write(f"{'#'*60}\n")
        out.write(content + "\n\n")

print("Regenerated digest_concept_core_to_0vram.txt successfully")
