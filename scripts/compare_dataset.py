import os


def collect_files_without_ext(root):
    file_map = {}
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            rel_dir = os.path.relpath(dirpath, root)
            base_name, ext = os.path.splitext(filename)
            rel_path = os.path.normpath(os.path.join(rel_dir, base_name))
            if rel_path not in file_map:
                file_map[rel_path] = set()
            file_map[rel_path].add(ext)
    return file_map


def compare_folders(folder1, folder2):
    files1 = collect_files_without_ext(folder1)
    files2 = collect_files_without_ext(folder2)

    all_keys = set(files1.keys()) | set(files2.keys())
    for key in sorted(all_keys):
        exts1 = files1.get(key, set())
        exts2 = files2.get(key, set())
        if len(exts1) > 1:
            print(f"Multiple extensions in folder1 for '{key}': {exts1}")
        elif len(exts2) > 1:
            print(f"Multiple extensions in folder2 for '{key}': {exts2}")
        elif exts1 == set():
            print(f"Missing file in folder1: '{key}'")
        elif exts2 == set():
            print(f"Missing file in folder2: '{key}'")
        elif exts1 != exts2:
            print(f"Difference at '{key}': {exts1} vs {exts2}")


# Example usage
compare_folders(
    "/home/jteng2127/work/dataset/AIFood_SingleFood_0313",
    "/home/jteng2127/work/dataset/AIFood_SingleFood_0313_resized",
)
