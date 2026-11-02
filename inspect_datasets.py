import os

dataset = "archive (4)"

for folder in ["Output_train", "Output_validation"]:
    folder_path = os.path.join(dataset, folder)

    print(f"\n--- {folder} ---")

    if not os.path.isdir(folder_path):
        print("Folder not found!")
        continue

    for filename in sorted(os.listdir(folder_path))[:5]:
        file_path = os.path.join(folder_path, filename)

        print("\nFile:", filename)

        if filename.lower().endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    print("Label content:", f.read(500))
            except Exception as e:
                print("Could not read:", e)