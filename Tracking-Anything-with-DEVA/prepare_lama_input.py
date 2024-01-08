from PIL import Image
import numpy as np
import os
import shutil
import cv2
import sys

# Check if the user provided an argument
if len(sys.argv) != 4:
    print("Usage: python3 {} <img_path> <mask_path> <lama_path>".format(sys.argv[0]))
    sys.exit(1)

dataset_name = sys.argv[1]

image_dir = sys.argv[1]
mask_dir = os.path.join(sys.argv[2],"Annotations")
out_dir = sys.argv[3]
out_mask_dir = os.path.join(sys.argv[3],"label")
out_mask_vis_dir = os.path.join(sys.argv[3],"label_vis")
os.makedirs(out_dir,exist_ok=True)
os.makedirs(out_mask_dir,exist_ok=True)
os.makedirs(out_mask_vis_dir,exist_ok=True)

print("Image dir:   ", image_dir)
print("Mask dir:   ", mask_dir)
print("Lama input dir:   ", out_dir)

for name in sorted(os.listdir(image_dir)):
    print(os.path.join(image_dir,name))
    shutil.copy(os.path.join(image_dir,name),os.path.join(out_dir,name))
    
    mask = cv2.imread(os.path.join(mask_dir,name))
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY_INV)
    binary_mask = cv2.bitwise_not(binary_mask)

    # You can change the mask dilation kernel size and dilated_iterations according to your dataset.
    kernel_size = 5 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_iterations = 5
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=dilated_iterations)  

    cv2.imwrite(os.path.join(out_mask_vis_dir,name), dilated_mask)
    dilated_mask[dilated_mask>0] = 1
    cv2.imwrite(os.path.join(out_mask_dir,name), dilated_mask)



