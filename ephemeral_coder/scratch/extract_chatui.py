with open('scratch/user_history.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

out = []
for i in range(1280, 1330):
    if i < len(lines):
        out.append(f"Line {i+1}: {lines[i].strip()}")

with open('scratch/chatui_history.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(out))
