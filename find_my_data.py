import os

# Let's see where we are currently standing
print(f"Current Working Directory: {os.getcwd()}")

# Let's list everything inside MiniProj
base_path = r"C:\Users\LENOVO\Desktop\MiniProj"

if os.path.exists(base_path):
    print(f"\nScanning {base_path}...")
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            if "metadata" in f or f.endswith(".csv"):
                print(f"{sub_indent}⭐ FOUND: {f}")
else:
    print("❌ The BASE_DIR path itself seems wrong.")