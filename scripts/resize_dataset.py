from PIL import Image
import os

from tqdm import tqdm

MAX_SIZE = 4160
MAX_SIZE = 1024

# def resize_image(src_path, dest_path):
#     # if os.path.exists(dest_path):
#     #     return False
#     result = True
#     with Image.open(src_path) as img:
#         w, h = img.size
#         if w <= MAX_SIZE and h <= MAX_SIZE:
#             result = False
#         else:
#             scale = min(MAX_SIZE / w, MAX_SIZE / h)
#             new_size = (int(w * scale), int(h * scale))
#             img = img.resize(new_size, Image.LANCZOS)
#             img = img.convert("RGB")  # Force 3-channel RGB
#         dest_path = os.path.splitext(dest_path)[0] + ".png"  # Ensure .png extension
#         os.makedirs(os.path.dirname(dest_path), exist_ok=True)
#         img.save(dest_path, format="PNG")
#     return result


def resize_image(src_path, dest_path):
    if os.path.exists(dest_path):
        return "exists"
    with Image.open(src_path) as img:
        w, h = img.size
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if w > MAX_SIZE or h > MAX_SIZE:
            save_path = os.path.splitext(dest_path)[0] + ".png"
            if os.path.exists(save_path):
                return "exists"
            scale = min(MAX_SIZE / w, MAX_SIZE / h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
            img.convert("RGB").save(save_path, format="PNG")
            return "resized"
        else:
            save_path = os.path.splitext(dest_path)[0] + f".{img.format.lower()}"
            if os.path.exists(save_path):
                return "exists"
            img.convert("RGB").save(save_path, format=img.format)
            return "saved"


def scan_and_resize(root_dir, dest_dir):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp")
    image_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(exts):
                image_files.append(os.path.join(root, f))

    resized_count = 0
    saved_count = 0
    for f in (
        pb := tqdm(
            image_files,
            total=len(image_files),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
    ):
        rel_path = os.path.relpath(f, root_dir)
        dest_path = os.path.join(dest_dir, rel_path)
        try:
            result = resize_image(f, dest_path)
            if result == "resized":
                resized_count += 1
            elif result == "saved":
                saved_count += 1
            pb.set_postfix_str(f"resized: {resized_count}, saved: {saved_count}")
        except Exception as e:
            tqdm.write(f"Failed to resize {f}: {e}")
    print(f"Resized {resized_count} images")


if __name__ == "__main__":
    print(f"Resizing images to max size {MAX_SIZE}x{MAX_SIZE}")
    # print("Resizing foodseg103 dataset")
    # scan_and_resize(
    #     root_dir="/home/jteng2127/work/dataset/foodseg103",
    #     dest_dir="/home/jteng2127/work/dataset/foodseg103_resized_2",
    # )
    print("Resizing AIFood_SingleFood_0313 dataset")
    scan_and_resize(
        root_dir="/home/jteng2127/work/dataset/AIFood_SingleFood_0313",
        dest_dir="/home/jteng2127/work/dataset/AIFood_SingleFood_0313_resized",
    )
    print("Done")
