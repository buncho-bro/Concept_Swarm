with open('scratch/user_history.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

targets = [578, 579, 580, 581, 582, 1038, 1039, 1040, 1041, 1042, 1393, 1394, 1395, 1396, 1400, 1401]
out = []
for t in targets:
    if t < len(lines):
        out.append(f"Line {t+1}: {lines[t].strip()}")

with open('scratch/history_inspect_utf8.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(out))
