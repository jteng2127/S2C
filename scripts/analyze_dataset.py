import os
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# 設定資料夾路徑
# root_dir = "/home/jteng2127/work/dataset/foodseg103_resized"
# root_dir = "/home/jteng2127/work/dataset/VOC2012/JPEGImages"
root_dir = "/home/jteng2127/work/dataset/AIFood_SingleFood_0313_resized"

# 支援的圖片副檔名
image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
image_exts_count = {}

widths = []
heights = []

# 遞迴走訪
image_paths = []
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        image_ext = os.path.splitext(filename)[1]
        if image_ext not in image_exts_count:
            image_exts_count[image_ext] = 0
        image_exts_count[image_ext] += 1
        if not filename.lower().endswith(image_exts):
            continue
        path = os.path.join(dirpath, filename)
        image_paths.append(path)

print("Image extensions count:")
for ext, count in image_exts_count.items():
    print(f"{ext}: {count}")

for i, path in tqdm(enumerate(image_paths), total=len(image_paths)):
    try:
        with Image.open(path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    except Exception as e:
        print(f"Error processing {path}: {e}")

    if i % (len(image_paths) // 10) == 0:
        plt.scatter(widths, heights, alpha=0.5, s=1)
        plt.title(f"Image Width vs Height ({i}/{len(image_paths)})")
        plt.xlabel("Width (px)")
        plt.ylabel("Height (px)")
        # plt.ylim(-220, 8500)
        # plt.xlim(-180, 7700)
        plt.grid(True)
        plt.savefig("image_size.png")
        plt.clf()

plt.scatter(widths, heights, alpha=0.5, s=1)
plt.title(f"Image Width vs Height (final)")
plt.xlabel("Width (px)")
plt.ylabel("Height (px)")
# plt.ylim(-220, 8500)
# plt.xlim(-180, 7700)
plt.grid(True)
plt.savefig("image_size.png")
plt.clf()
