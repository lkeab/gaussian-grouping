import os
import numpy as np
from PIL import Image
import cv2
import sys

dataset_name = sys.argv[1]

gt_folder_path = os.path.join('data/lerf_mask',dataset_name,'test_mask')
# You can change pred_folder_path to your output
pred_folder_path = os.path.join('result/lerf_mask',dataset_name)

# General util function to get the boundary of a binary mask.
# https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    dt = (dt>128).astype('uint8')
    gt = (gt>128).astype('uint8')
    

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


def load_mask(mask_path):
    """Load the mask from the given path."""
    if os.path.exists(mask_path):
        return np.array(Image.open(mask_path).convert('L'))  # Convert to grayscale
    return None

def resize_mask(mask, target_shape):
    """Resize the mask to the target shape."""
    return np.array(Image.fromarray(mask).resize((target_shape[1], target_shape[0]), resample=Image.NEAREST))

def calculate_iou(mask1, mask2):
    """Calculate IoU between two boolean masks."""
    mask1_bool = mask1 > 128
    mask2_bool = mask2 > 128
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    iou = np.sum(intersection) / np.sum(union)
    return iou

iou_scores = {}  # Store IoU scores for each class
biou_scores = {}
class_counts = {}  # Count the number of times each class appears

# Iterate over each image and category in the GT dataset
for image_name in os.listdir(gt_folder_path):
    gt_image_path = os.path.join(gt_folder_path, image_name)
    pred_image_path = os.path.join(pred_folder_path, image_name)
    
    if os.path.isdir(gt_image_path):
        for cat_file in os.listdir(gt_image_path):
            cat_id = cat_file.split('.')[0]  # Assuming cat_file format is "cat_id.png"
            gt_mask_path = os.path.join(gt_image_path, cat_file)
            pred_mask_path = os.path.join(pred_image_path, cat_file)

            gt_mask = load_mask(gt_mask_path)
            pred_mask = load_mask(pred_mask_path)
            print("GT:  ",gt_mask_path)
            print("Pred:  ",pred_mask_path)

            if gt_mask is not None and pred_mask is not None:
                # Resize prediction mask to match GT mask shape if they are different
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = resize_mask(pred_mask, gt_mask.shape)

                iou = calculate_iou(gt_mask, pred_mask)
                biou = boundary_iou(gt_mask, pred_mask)
                print("IoU: ",iou," BIoU:   ",biou)
                if cat_id not in iou_scores:
                    iou_scores[cat_id] = []
                    biou_scores[cat_id] = []
                iou_scores[cat_id].append(iou)
                biou_scores[cat_id].append(biou)
                class_counts[cat_id] = class_counts.get(cat_id, 0) + 1

# Calculate mean IoU for each class
mean_iou_per_class = {cat_id: np.mean(iou_scores[cat_id]) for cat_id in iou_scores}
mean_biou_per_class = {cat_id: np.mean(biou_scores[cat_id]) for cat_id in biou_scores}

# Calculate overall mean IoU
overall_mean_iou = np.mean(list(mean_iou_per_class.values()))
overall_mean_biou = np.mean(list(mean_biou_per_class.values()))

print("Mean IoU per class:", mean_iou_per_class)
print("Mean Boundary IoU per class:", mean_biou_per_class)
print("Overall Mean IoU:", overall_mean_iou)
print("Overall Boundary Mean IoU:", overall_mean_biou)