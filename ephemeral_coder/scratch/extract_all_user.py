import json
import re

transcript_path = r"C:\Users\lop3l\.gemini\antigravity\brain\126586e2-f977-4633-9d33-4d5f6ad332c5\.system_generated\logs\transcript.jsonl"
out_path = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\user_history.txt"

with open(transcript_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
    for line_idx, line in enumerate(f):
        data = json.loads(line)
        if data.get("type") == "USER_INPUT":
            content = data.get("content", "").strip()
            # clean USER_REQUEST tags
            m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.DOTALL)
            req = m.group(1).strip() if m else content
            out.write(f"=== [Step {data.get('step_index')}] (Line {line_idx+1}) ===\n")
            out.write(req + "\n\n")

print("Done extracting user inputs to user_history.txt")
