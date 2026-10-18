import re

history_path = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\filtered_history.txt"
out_path = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\timeline_summary.txt"

with open(history_path, "r", encoding="utf-8") as f:
    text = f.read()

sections = text.split("==================================================")

key_steps = [
    0, 84, 154, 204, 234, 341, 352, 380, 904, 1393, 2416, 2607, 2672, 2757, 2837,
    2852, 2956, 2995, 3019, 3091, 3230, 3248, 3338, 3460, 3492, 3559, 3567, 3659,
    3663, 3732, 3737, 3882, 4157, 4169, 4191, 4203, 4396, 4501, 4512, 4608, 4708,
    4776, 4827, 4943, 6045
]

with open(out_path, "w", encoding="utf-8") as out:
    for sec in sections:
        if "=== STEP " in sec:
            lines = sec.strip().splitlines()
            header = lines[0]
            m = re.search(r"STEP (\d+)", header)
            if m:
                s_num = int(m.group(1))
                # write out content of key steps
                out.write(f"########################################\n")
                out.write(f"### STEP {s_num}\n")
                out.write(f"########################################\n")
                body = "\n".join(lines[2:]).strip()
                out.write(body + "\n\n")

print("Done writing timeline_summary.txt")
