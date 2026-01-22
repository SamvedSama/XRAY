import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes


from scipy.spatial.distance import cdist

def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    return (2.0 * intersection) / (np.sum(gt) + np.sum(pred) + 1e-8)

def jaccard_index(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    return intersection / (union + 1e-8)

def hausdorff95(gt, pred):
    """
    Fast and stable HD95 using only boundary points.
    """
    # Find boundaries
    gt_edges = morphology.binary_dilation(gt) ^ gt
    pred_edges = morphology.binary_dilation(pred) ^ pred

    gt_pts = np.column_stack(np.where(gt_edges > 0))
    pred_pts = np.column_stack(np.where(pred_edges > 0))

    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return np.nan

    # Distance matrix between boundary points (~1k x 1k)
    D = cdist(gt_pts, pred_pts)

    d1 = np.min(D, axis=1)
    d2 = np.min(D, axis=0)

    return np.percentile(np.hstack([d1, d2]), 95)



# -------------------------------------------------
# Image Reading & Preprocessing
# -------------------------------------------------
def read_and_preprocess(path, size=(600, 600)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    return img.astype(np.uint8)


# -------------------------------------------------
# Euler-number–based Binary Extraction (FIXED)
# -------------------------------------------------
def euler_binary_extraction(img):
    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary = (binary > 0).astype(np.uint8)

    labels = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels)

    cleaned = np.zeros_like(binary)

    for region in props:
        if region.area > 5000 and region.euler_number <= 0:
            cleaned[labels == region.label] = 1

    cleaned = morphology.binary_closing(cleaned, morphology.disk(5))
    cleaned = morphology.binary_opening(cleaned, morphology.disk(3))

    return cleaned.astype(np.uint8)


# -------------------------------------------------
# Edge Detection & Morphology
# -------------------------------------------------
def canny_edge(binary):
    return cv2.Canny(binary * 255, 50, 150)


def morphological_refinement(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(edges, kernel, iterations=2)
    ero = cv2.erode(dil, kernel, iterations=1)
    return ero


# -------------------------------------------------
# Region Filling & Lung Extraction
# -------------------------------------------------
def fill_lung_regions(edge_img):
    return binary_fill_holes(edge_img > 0).astype(np.uint8)


def extract_lungs(filled_img):
    labels = measure.label(filled_img, connectivity=2)
    props = sorted(measure.regionprops(labels), key=lambda x: x.area, reverse=True)

    lung_mask = np.zeros_like(filled_img)
    for region in props[:2]:
        lung_mask[labels == region.label] = 1

    lung_mask = morphology.binary_closing(lung_mask, morphology.disk(10))
    lung_mask = morphology.binary_opening(lung_mask, morphology.disk(5))
    return lung_mask.astype(np.uint8)


# -------------------------------------------------
# Segmentation Pipeline
# -------------------------------------------------
def lung_segmentation_pipeline_from_image(img):
    binary = euler_binary_extraction(img)
    edges = canny_edge(binary)
    morph = morphological_refinement(edges)
    filled = fill_lung_regions(morph)
    lung_mask = extract_lungs(filled)
    segmented = img * lung_mask
    return lung_mask, segmented


# -------------------------------------------------
# Ground Truth Mask Loader
# -------------------------------------------------
def read_mask(path, size=(600, 600)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")

    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


# -------------------------------------------------
# Visualization
# -------------------------------------------------
def save_comparison(img, gt_mask, pred_mask, title, save_path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original CXR")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Predicted")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------
# GENERALIZED DATASET PROCESSOR
# -------------------------------------------------
def process_dataset(dataset_name, images_path, masks_path=None):
    print(f"\n=== Processing {dataset_name} Dataset ===")

    out_dir = f"outputs/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(images_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    # --------------------------
    # METRIC STORAGE
    # --------------------------
    dice_vals = []
    jacc_vals = []
    hd95_vals = []

    # Open metrics file for logging
    metrics_path = os.path.join(out_dir, "metrics.txt")
    metrics_file = open(metrics_path, "w")
    metrics_file.write(f"=== METRICS FOR DATASET: {dataset_name} ===\n\n")

    for fname in image_files:
        img_path = os.path.join(images_path, fname)
        img = read_and_preprocess(img_path)

        pred_mask, _ = lung_segmentation_pipeline_from_image(img)

        if masks_path is not None:
            mask_path = os.path.join(masks_path, fname)

            if not os.path.exists(mask_path):
                print(f"[WARNING] Missing GT for {fname}")
                metrics_file.write(f"{fname}: GT MISSING\n")
                continue

            gt_mask = read_mask(mask_path)

            # --------------------------
            # COMPUTE METRICS
            # --------------------------
            dsc = dice_score(gt_mask, pred_mask)
            jacc = jaccard_index(gt_mask, pred_mask)
            hd = hausdorff95(gt_mask, pred_mask)

            dice_vals.append(dsc)
            jacc_vals.append(jacc)
            hd95_vals.append(hd)

            # Write to text file
            metrics_file.write(
                f"{fname}: DSC={dsc:.4f}, IoU={jacc:.4f}, HD95={hd:.2f}\n"
            )

            print(f"{fname}:  DSC={dsc:.4f}  IoU={jacc:.4f}  HD95={hd:.2f}")

            save_path = os.path.join(out_dir, f"{dataset_name}_{fname}")
            save_comparison(img, gt_mask, pred_mask,
                            f"{dataset_name}: {fname}", save_path)

        else:
            save_path = os.path.join(out_dir, f"{dataset_name}_{fname}")
            plt.imsave(save_path, pred_mask, cmap='gray')
            metrics_file.write(f"{fname}: NO GT (Prediction saved)\n")

        print(f"[SAVED] {save_path}")

    # --------------------------
    # DATASET-LEVEL AVERAGES
    # --------------------------
    if masks_path is not None:
        mean_dice = np.nanmean(dice_vals)
        mean_jacc = np.nanmean(jacc_vals)
        mean_hd95 = np.nanmean(hd95_vals)

        print("\n=== AVERAGE METRICS FOR:", dataset_name, "===")
        print(f"Mean Dice:      {mean_dice:.4f}")
        print(f"Mean Jaccard:   {mean_jacc:.4f}")
        print(f"Mean HD95:      {mean_hd95:.2f}")
        print("============================================")

        # Write dataset averages to metrics file
        metrics_file.write("\n--- AVERAGE METRICS ---\n")
        metrics_file.write(f"Mean DSC: {mean_dice:.4f}\n")
        metrics_file.write(f"Mean IoU: {mean_jacc:.4f}\n")
        metrics_file.write(f"Mean HD95: {mean_hd95:.2f}\n")

    metrics_file.close()
    print(f"[METRICS SAVED] {metrics_path}")

    # --------------------------
    # DATASET-LEVEL AVERAGES
    # --------------------------
    if masks_path is not None:
        print("\n=== AVERAGE METRICS FOR:", dataset_name, "===")
        print(f"Mean Dice:      {np.nanmean(dice_vals):.4f}")
        print(f"Mean Jaccard:   {np.nanmean(jacc_vals):.4f}")
        print(f"Mean HD95:      {np.nanmean(hd95_vals):.2f}")
        print("============================================")



# -------------------------------------------------
# RUN ALL DATASETS
# -------------------------------------------------
process_dataset("COVIDQU",
                images_path="Used/COVIDQU/lung",
                masks_path="Used/COVIDQU/mask")

process_dataset("JSRT",
                images_path="Used/JSRT/lung",
                masks_path="Used/JSRT/mask")

process_dataset("PA",
                images_path="Used/pa/lung",
                masks_path="Used/pa/mask")

process_dataset("SYNTH",
                images_path="Used/synth",
                masks_path=None)
