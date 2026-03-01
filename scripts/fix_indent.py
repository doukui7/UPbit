import os

file_path = r'c:\Users\박두규\Desktop\upbit\src\ui\coin_mode.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Remove one level of indentation (4 spaces) if possible, starting from line after def
    if line.startswith('def render_coin_mode'):
        new_lines.append(line)
        continue
    
    if line.startswith('    '):
        new_lines.append(line[4:])
    else:
        new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fixed indentation in src/ui/coin_mode.py")
