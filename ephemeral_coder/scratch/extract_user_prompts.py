import json

transcript_path = r"C:\Users\lop3l\.gemini\antigravity\brain\126586e2-f977-4633-9d33-4d5f6ad332c5\.system_generated\logs\transcript.jsonl"

keywords = ["概念核", "0VRAM", "0VARM", "蒸留"]

with open(transcript_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        data = json.loads(line)
        content = data.get("content", "")
        # check user inputs or model responses containing keywords
        if data.get("type") == "USER_INPUT":
            for kw in keywords:
                if kw in content:
                    print(f"[Line {line_idx+1}] [USER]: {content[:200]}...\n")
                    break
