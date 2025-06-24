import os

val_dir = 'val'  # Change this to your actual path

# List all class folders
folders = [f for f in os.listdir(val_dir) if f.startswith("class")]

# Sort folders in descending order to avoid overwriting
folders_sorted = sorted(folders, key=lambda x: int(x.replace("class", "")), reverse=False)

for folder in folders_sorted:
    old_path = os.path.join(val_dir, folder)
    class_num = int(folder.replace("class", ""))
    new_folder = f"class{class_num - 1}"
    new_path = os.path.join(val_dir, new_folder)

    print(f"Renaming: {old_path} -> {new_path}")
    os.rename(old_path, new_path)


# val_dir = 'val'  # Change this to your actual path

# # List all class folders
# folders = [f for f in os.listdir(val_dir) if f.startswith("class")]

# # Sort folders in descending order to avoid overwriting
# folders_sorted = sorted(folders, key=lambda x: int(x.replace("class", "")), reverse=True)

# for folder in folders_sorted:
#     old_path = os.path.join(val_dir, folder)
#     class_num = int(folder.replace("class", ""))
#     new_folder = f"class{class_num + 1}"
#     new_path = os.path.join(val_dir, new_folder)

#     print(f"Renaming: {old_path} -> {new_path}")
#     os.rename(old_path, new_path)
