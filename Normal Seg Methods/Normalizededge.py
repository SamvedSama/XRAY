import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import cdist


# -------------------------------------------------
# METRIC FUNCTIONS
# -------------------------------------------------
def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    return (2.0 * intersection) / (np.sum(gt) + np.sum(pred) + 1e-8)

def jaccard_index(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    return intersection / (union + 1e-8)

def hausdorff95(gt, pred):
    gt_edges = morphology.binary_dilation(gt) ^ gt
    pred_edges = morphology.binary_dilation(pred) ^ pred

    gt_pts = np.column_stack(np.where(gt_edges > 0))
    pred_pts = np.column_stack(np.where(pred_edges > 0))

    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return np.nan

    D = cdist(gt_pts, pred_pts)
    d1 = np.min(D, axis=1)
    d2 = np.min(D, axis=0)

    return np.percentile(np.hstack([d1, d2]), 95)


# -------------------------------------------------
# PREPROCESSING (CLAHE + Gaussian + Otsu)
# -------------------------------------------------
def read_and_preprocess(path, size=(600, 600)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Gaussian Blur (reduce noise before thresholding)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Normalize
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


# -------------------------------------------------
# Euler-based Extraction
# -------------------------------------------------
def euler_binary_extraction(img):
    _, binary = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (binary > 0).astype(np.uint8)

    labels = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels)

    cleaned = np.zeros_like(binary)
    for r in props:
        if r.area > 5000 and r.euler_number <= 0:
            cleaned[labels == r.label] = 1

    cleaned = morphology.binary_closing(cleaned, morphology.disk(5))
    cleaned = morphology.binary_opening(cleaned, morphology.disk(3))
    return cleaned.astype(np.uint8)


# -------------------------------------------------
# EDGE EXTRACTION + MORPHOLOGY
# -------------------------------------------------
def canny_edge(binary):
    return cv2.Canny(binary * 255, 50, 150)

def morphological_refinement(e):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    d = cv2.dilate(e, kernel, iterations=2)
    e = cv2.erode(d, kernel, iterations=1)
    return e


# -------------------------------------------------
# HOLE FILLING + LUNG EXTRACTION
# -------------------------------------------------
def fill_lung_regions(edge_img):
    return binary_fill_holes(edge_img > 0).astype(np.uint8)

def extract_lungs(filled):
    labels = measure.label(filled, connectivity=2)
    props = sorted(measure.regionprops(labels),
                   key=lambda x: x.area, reverse=True)

    lung_mask = np.zeros_like(filled)
    for r in props[:2]:
        lung_mask[labels == r.label] = 1

    lung_mask = morphology.binary_closing(lung_mask, morphology.disk(10))
    lung_mask = morphology.binary_opening(lung_mask, morphology.disk(5))
    return lung_mask.astype(np.uint8)


# -------------------------------------------------
# MAIN LUNG SEGMENTATION PIPELINE
# -------------------------------------------------
def lung_segmentation_pipeline_from_image(img):
    binary = euler_binary_extraction(img)
    edges = canny_edge(binary)
    morph = morphological_refinement(edges)
    filled = fill_lung_regions(morph)
    lung_mask = extract_lungs(filled)
    return lung_mask, img * lung_mask


# -------------------------------------------------
# Mask Reader
# -------------------------------------------------
def read_mask(path, size=(600, 600)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


# -------------------------------------------------
# SAVE COMPARISON IMAGE
# -------------------------------------------------
def save_comparison(img, gt, pred, title, path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# -------------------------------------------------
# MAIN PROCESS FOR ANY DATASET
# -------------------------------------------------
def process_dataset(name, img_dir, mask_dir=None):
    print(f"\n=== Processing {name} ===")

    # NEW: Main output folder
    base_out_dir = os.path.join("outputs_normal_edge", name)
    os.makedirs(base_out_dir, exist_ok=True)

    # NEW: Folder for predicted masks
    pred_mask_dir = os.path.join(base_out_dir, "masks")
    os.makedirs(pred_mask_dir, exist_ok=True)

    # Metrics file inside same folder
    metrics_path = os.path.join(base_out_dir, "metrics.txt")
    metrics_file = open(metrics_path, "w")
    metrics_file.write(f"=== METRICS FOR {name} ===\n\n")

    dice_vals = []
    jacc_vals = []
    hd_vals = []

    files = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    for fname in files:
        img_path = os.path.join(img_dir, fname)
        img = read_and_preprocess(img_path)

        pred, _ = lung_segmentation_pipeline_from_image(img)

        # -------------------------------------------------
        # SAVE PREDICTED MASK ALWAYS
        # -------------------------------------------------
        plt.imsave(os.path.join(pred_mask_dir, fname), pred, cmap="gray")

        if mask_dir:  # has GT
            mask_path = os.path.join(mask_dir, fname)

            if not os.path.exists(mask_path):
                metrics_file.write(f"{fname}: GT missing\n")
                continue

            gt = read_mask(mask_path)

            # Metrics
            d = dice_score(gt, pred)
            j = jaccard_index(gt, pred)
            h = hausdorff95(gt, pred)

            dice_vals.append(d)
            jacc_vals.append(j)
            hd_vals.append(h)

            metrics_file.write(f"{fname}: DSC={d:.4f}, IoU={j:.4f}, HD95={h:.2f}\n")

        else:
            metrics_file.write(f"{fname}: Mask saved (No GT)\n")

        print("[DONE]", fname)

    # Averages
    if mask_dir:
        metrics_file.write("\n--- AVERAGE METRICS ---\n")
        metrics_file.write(f"Mean DSC:  {np.nanmean(dice_vals):.4f}\n")
        metrics_file.write(f"Mean IoU:  {np.nanmean(jacc_vals):.4f}\n")
        metrics_file.write(f"Mean HD95: {np.nanmean(hd_vals):.2f}\n")

    metrics_file.close()

    print(f"All outputs for {name} saved in: {base_out_dir}/")



# -------------------------------------------------
# RUN ON NEW DATASET STRUCTURE
# -------------------------------------------------
process_dataset(
    "JSRT",
    img_dir="Normalized Data/jsrt/cxr",
    mask_dir="Normalized Data/jsrt/masks"
)

process_dataset(
    "SYNTH",
    img_dir="Normalized Data/synth/cxr",
    mask_dir=None
)
