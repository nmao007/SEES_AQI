import os

print("==================================================")
print("1. Where Python is currently looking from:")
print(f"   Current Working Directory (CWD): {os.getcwd()}")
print("==================================================")

target_folder = "data/processed_csvs/data_publicCSV"

print(f"2. Checking target directory: '{target_folder}'")
if os.path.exists(target_folder):
    print("   [SUCCESS] Directory found! Here are the EXACT file names inside:")
    print("   --------------------------------------------------------")
    for file_name in sorted(os.listdir(target_folder)):
        print(f"   - '{file_name}'")
    print("   --------------------------------------------------------")
    print("   👉 Compare these exact names (including uppercase/lowercase and spaces)")
    print("      with the string variables at the top of your data_processing.py file!")
else:
    print(f"   [ERROR] The path '{target_folder}' does NOT exist from here.")
    print("   This means your terminal is not running from the root folder you think it is.")
    
    # Let's see what folders DO exist here to help you find your way
    print("\n3. Available folders in your current directory:")
    print(os.listdir('.'))
print("==================================================")