import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from collections import defaultdict

import numpy as np
import torch

# Custom
from tools.imutils import *

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from dotenv import load_dotenv
load_dotenv()

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()  # Pause execution until debugger is attached

# paths
root_path = '.'
sam_path = os.path.join(os.getenv('PRETRAINED_DIR'), 'sam_vit_h.pth')
img_path = os.path.join(os.getenv('VOC2012_DIR'), 'JPEGImages')
save_path = os.getenv('SE_DIR')
os.makedirs(save_path, exist_ok=True)

print(f'img_path: {img_path}')
print(f'save_path: {save_path}')

# load image names
img_list_path = os.path.join(root_path, 'voc12', 'train_aug.txt')
img_gt_name_list = open(img_list_path).read().splitlines()
img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

# create sam models
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise ValueError('No GPU available!')

sam_models = []
mask_generators = []
for i in range(num_gpus):
    sam = sam_model_registry['vit_h'](checkpoint=sam_path).to(f'cuda:{i}')
    sam_models.append(sam)
    mask_generators.append(SamAutomaticMaskGenerator(sam))
print('SAM models created')

    
def process_image(name, mask_generator_queue: Queue):
    mask_generator: SamAutomaticMaskGenerator = mask_generator_queue.get(timeout=1)
    device = str(mask_generator.predictor.device)

    img_path_full = os.path.join(img_path, name + '.jpg')
    img = cv2.imread(img_path_full)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(img)

    temp = np.full((img.shape[0], img.shape[1]), -1, dtype=int)
    for i, mask in enumerate(reversed(masks)):
        temp[mask['segmentation']] = i

    np.save(os.path.join(save_path, name), temp)

    mask_generator_queue.put(mask_generator)
    return name, device

if __name__ == "__main__":
    mask_generator_queue = Queue()
    for mask_generator in mask_generators:
        mask_generator_queue.put(mask_generator)
    mask_generator_count = defaultdict(int)

    print('Assigning tasks...')
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for name in tqdm(img_name_list, total=len(img_name_list)):
            futures.append(executor.submit(process_image, name, mask_generator_queue))
        # futures = [executor.submit(process_image, name, mask_generator_queue) for name in img_name_list]

        print('Processing images...')
        for future in (pbar := tqdm(as_completed(futures), total=len(futures))):
            image_name, device = future.result()
            mask_generator_count[device] += 1
            pbar.set_postfix({"img": image_name, **mask_generator_count})