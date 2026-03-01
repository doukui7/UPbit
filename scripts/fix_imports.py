import os
import re

root_dir = r'c:\Users\박두규\Desktop\upbit'
src_dir = os.path.join(root_dir, 'src')

replacements = [
    (r'\bfrom strategy\b', 'from src.strategy'),
    (r'\bimport strategy\b', 'import src.strategy'),
    (r'\bfrom backtest\b', 'from src.backtest'),
    (r'\bimport backtest\b', 'import src.backtest'),
    (r'\bfrom trading\b', 'from src.trading'),
    (r'\bimport trading\b', 'import src.trading'),
    (r'\bfrom kis_trader\b', 'from src.engine.kis_trader'),
    (r'\bimport kis_trader\b', 'import src.engine.kis_trader'),
    (r'\bfrom data_cache\b', 'from src.engine.data_cache'),
    (r'\bimport data_cache\b', 'import src.engine.data_cache'),
    (r'\bfrom data_manager\b', 'from src.engine.data_manager'),
    (r'\bimport data_manager\b', 'import src.engine.data_manager'),
    (r'\bfrom kiwoom_gold\b', 'from src.engine.kiwoom_gold'),
    (r'\bimport kiwoom_gold\b', 'import src.engine.kiwoom_gold'),
]

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for pattern, repl in replacements:
        new_content = re.sub(pattern, repl, new_content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

# Files to process: app.py and everything in src/
files_to_process = [os.path.join(root_dir, 'app.py')]
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith('.py'):
            files_to_process.append(os.path.join(root, file))

modified_count = 0
for file_path in files_to_process:
    if process_file(file_path):
        modified_count += 1
        print(f"Modified: {file_path}")

print(f"Total modified files: {modified_count}")
