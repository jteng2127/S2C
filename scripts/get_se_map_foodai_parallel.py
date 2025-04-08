import os
import traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from collections import defaultdict

import numpy as np
import torch

from tools.imutils import *
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from dotenv import load_dotenv
from tools.dataset import FoodDataset

load_dotenv()

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()  # Pause execution until debugger is attached

# paths
root_path = "."
sam_path = os.path.join(os.getenv("PRETRAINED_DIR"), "sam_vit_h.pth")
dataset_path = os.path.join(os.getenv("AIFOOD_SINGLEFOOD_DIR"))
save_path = os.getenv("AIFOOD_SINGLEFOOD_SE_DIR")
os.makedirs(save_path, exist_ok=True)

print(f"dataset_path: {dataset_path}")
print(f"save_path: {save_path}")

# load image names
train_csv_path = os.path.join(dataset_path, "AllFoodImage_train_ratio811.csv")
val_csv_path = os.path.join(dataset_path, "AllFoodImage_valid_ratio811.csv")
test_csv_path = os.path.join(dataset_path, "AllFoodImage_test_ratio811.csv")
train_dataset = FoodDataset(root=train_csv_path)
val_dataset = FoodDataset(root=val_csv_path)
test_dataset = FoodDataset(root=test_csv_path)

print("Datasets loaded")

img_filename_list = []
for dataset in [train_dataset, val_dataset, test_dataset]:
    for img_data in dataset.datalist:
        img_filename_list.append(img_data[0])
# img_filename_list = img_filename_list[:100]
# img_filename_list = [
#     filename
#     for filename in img_filename_list
#     if filename == "6_Fruit/H_Fruits/H1_FreshFruits/Kiwi/Kiwi_13.jpeg"
# ]

# create sam models
num_gpus = torch.cuda.device_count()
print(f"num_gpus: {num_gpus}")
if num_gpus == 0:
    raise ValueError("No GPU available!")

sam_models = []
mask_generators = []
for i in range(num_gpus):
    sam = sam_model_registry["vit_h"](checkpoint=sam_path).to(f"cuda:{i}")
    sam_models.append(sam)
    mask_generators.append(SamAutomaticMaskGenerator(sam))
print("SAM models created")


def process_image(filename, mask_generator_queue: Queue):
    try:
        mask_generator: SamAutomaticMaskGenerator = mask_generator_queue.get(timeout=1)
        device = str(mask_generator.predictor.device)

        npy_filepath = os.path.join(save_path, filename)
        npy_filepath = os.path.splitext(npy_filepath)[0] + ".npy"

        if not os.path.exists(npy_filepath) or True:
            img_path_full = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path_full)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            s = 4160
            # if w or h is larger that 4300, resize it proportionally
            if img.shape[0] > 4160 or img.shape[1] > 4160:
                if img.shape[0] > img.shape[1]:
                    img = cv2.resize(
                        img, (int(4160 * img.shape[1] / img.shape[0]), 4160)
                    )
                else:
                    img = cv2.resize(
                        img, (4160, int(4160 * img.shape[0] / img.shape[1]))
                    )

            masks = mask_generator.generate(img)

            img = np.full((img.shape[0], img.shape[1]), -1, dtype=int)
            for i, mask in enumerate(reversed(masks)):
                img[mask["segmentation"]] = i

            os.makedirs(os.path.dirname(npy_filepath), exist_ok=True)
            np.save(npy_filepath, img)

            del img, masks
            torch.cuda.empty_cache()

        mask_generator_queue.put(mask_generator)
        print(f"Processed {filename} on {device}")
        return filename, device
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        traceback.print_exc()
        raise e
    finally:
        mask_generator_queue.put(mask_generator)


if __name__ == "__main__":
    mask_generator_queue = Queue()
    for mask_generator in mask_generators:
        mask_generator_queue.put(mask_generator)
    # mask_generator_count = defaultdict(int)

    print("Assigning tasks...")
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for filename in tqdm(img_filename_list, total=len(img_filename_list)):
            futures.append(
                executor.submit(process_image, filename, mask_generator_queue)
            )
        # futures = [executor.submit(process_image, name, mask_generator_queue) for name in img_name_list]

        print("Processing images...")
        with tqdm(total=len(futures)) as pbar:
            for i, future in enumerate(as_completed(futures)):
                try:
                    image_name, device = future.result()
                    pbar.update()
                except Exception as e:
                    print(f"Error processing {i}: {e}")
                # mask_generator_count[device] += 1
                # pbar.set_postfix({"img": image_name, **mask_generator_count})
