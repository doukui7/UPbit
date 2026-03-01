import os
import sys
import subprocess
import py_compile

def run_check():
    print("=== Starting CI Stability Check ===")
    
    error_found = False
    
    # 1. Syntax Check
    print("\n[1/2] Checking Syntax (py_compile)...")
    src_files = []
    for root, _, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                src_files.append(os.path.join(root, file))
    src_files.append('app.py')
    
    for file in src_files:
        try:
            py_compile.compile(file, doraise=True)
            print(f" OK: {file}")
        except py_compile.PyCompileError as e:
            print(f" ERROR: {file}\n{e}")
            error_found = True

    # 2. Ruff Lint Check (if installed)
    print("\n[2/2] Checking Lint (ruff)...")
    try:
        result = subprocess.run(['ruff', 'check', '.'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f" LINT WARNINGS/ERRORS:\n{result.stdout}")
            # we don't necessarily fail CI for linting unless required
        else:
            print(" OK: No lint issues found.")
    except FileNotFoundError:
        print(" SKIP: ruff not found in environment.")

    if error_found:
        print("\n[FAIL] CI Check Failed!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] CI Check Passed!")
        sys.exit(0)

if __name__ == "__main__":
    run_check()
