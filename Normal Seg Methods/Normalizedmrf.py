import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import measure, morphology
from sklearn.mixture import GaussianMixture
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import cdist
import maxflow


# -------------------------------------------------
# METRICS
# -------------------------------------------------
def dice_score(gt, pred):
    inter = np.sum(gt * pred)
    return (2 * inter) / (np.sum(gt) + np.sum(pred) + 1e-8)


def jaccard_index(gt, pred):
    inter = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - inter
    return inter / (union + 1e-8)


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

    return np.percentile(np.concatenate([d1, d2]), 95)


# -------------------------------------------------
# PREPROCESSING (CLAHE + Gaussian + Normalize)
# -------------------------------------------------
def read_and_preprocess(path, size=(600, 600)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    return img.astype(np.uint8)


def read_mask(path, size=(600, 600)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")

    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


# -------------------------------------------------
# EULER + EDGE-BASED LUNG PRIOR
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


def get_lung_prior(img):

    euler_mask = euler_binary_extraction(img)

    edges = cv2.Canny(euler_mask * 255, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    filled = binary_fill_holes(edges > 0).astype(np.uint8)

    labels = measure.label(filled, connectivity=2)
    props = sorted(measure.regionprops(labels), key=lambda x: x.area, reverse=True)

    lung_mask = np.zeros_like(filled)
    for r in props[:2]:
        lung_mask[labels == r.label] = 1

    lung_mask = morphology.binary_closing(lung_mask, morphology.disk(10))
    lung_mask = morphology.binary_opening(lung_mask, morphology.disk(5))

    return lung_mask.astype(np.uint8)


# -------------------------------------------------
# MRF SEGMENTATION (using lung prior)
# -------------------------------------------------
def mrf_lung_segmentation(img, lambda_smooth=20):

    lung_prior = get_lung_prior(img)

    if np.sum(lung_prior) < 10000:
        lung_prior = np.ones_like(img)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enh = clahe.apply(img)
    enh_norm = enh.astype(np.float32) / 255.0

    lung_pixels = enh_norm[lung_prior == 1].reshape(-1, 1)
    if len(lung_pixels) < 1000:
        lung_pixels = enh_norm.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2)
    gmm.fit(lung_pixels)

    means = gmm.means_.flatten()
    lung_label = np.argmin(means)
    bg_label = 1 - lung_label

    h, w = img.shape
    probs = gmm.predict_proba(enh_norm.reshape(-1, 1)).reshape(h, w, 2)

    data_lung = -np.log(probs[:, :, lung_label] + 1e-8)
    data_bg = -np.log(probs[:, :, bg_label] + 1e-8)

    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((h, w))
    g.add_grid_tedges(nodes, data_lung, data_bg)
    g.add_grid_edges(nodes, lambda_smooth)
    g.maxflow()

    seg = (~g.get_grid_segments(nodes)).astype(np.uint8)

    seg = seg * lung_prior

    # -------------------------------------------------
    # POST-PROCESSING (holes + smoothing + fill + 2 lungs)
    # -------------------------------------------------

    labels = measure.label(seg, connectivity=2)
    props = sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)

    final = np.zeros_like(seg)
    for r in props[:2]:
        final[labels == r.label] = 1

    final = morphology.remove_small_holes(final.astype(bool), area_threshold=8000)
    final = final.astype(np.uint8)

    final = morphology.binary_closing(final, morphology.disk(25))
    final = morphology.binary_opening(final, morphology.disk(10))

    labels = measure.label(final, connectivity=2)
    props = sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)

    clean = np.zeros_like(final)
    for r in props[:2]:
        clean[labels == r.label] = 1

    final = clean

    final = morphology.binary_closing(final, morphology.disk(8))

    # -------------------------------------------------
    # NEW: FULL REGION FILLING — SOLID LUNGS
    # -------------------------------------------------
    final = binary_fill_holes(final).astype(np.uint8)

    # Keep only top 2 again (safety)
    labels = measure.label(final, connectivity=2)
    props = sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)

    clean2 = np.zeros_like(final)
    for r in props[:2]:
        clean2[labels == r.label] = 1

    final = clean2

    return final.astype(np.uint8)


# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------
def save_with_gt(img, gt_mask, pred_mask, save_path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(gt_mask, cmap="gray"); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(pred_mask, cmap="gray"); plt.title("MRF Prediction"); plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_without_gt(img, pred_mask, save_path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(pred_mask, cmap="gray"); plt.title("MRF Prediction"); plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------------------------------------
# DATASET PROCESSOR
# -------------------------------------------------
def process_dataset(dataset_name, images_path, masks_path=None):

    print(f"\n=== Processing {dataset_name} ===")

    base_dir = os.path.join("outputs_normal_mrf", dataset_name)
    masks_dir = os.path.join(base_dir, "masks")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Saving to:", os.path.abspath(base_dir))

    metrics_file = open(os.path.join(base_dir, "metrics.txt"), "w")
    metrics_file.write(f"=== METRICS FOR {dataset_name} ===\n\n")

    dice_vals, jacc_vals, hd_vals = [], [], []

    img_files = sorted(f for f in os.listdir(images_path)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))

    for fname in img_files:
        img = read_and_preprocess(os.path.join(images_path, fname))
        pred_mask = mrf_lung_segmentation(img)

        plt.imsave(os.path.join(masks_dir, fname), pred_mask, cmap="gray")

        viz_path = os.path.join(results_dir, f"{fname}_MRF.png")

        if masks_path is None:
            save_without_gt(img, pred_mask, viz_path)
            metrics_file.write(f"{fname}: NO GT\n")
            continue

        gt = read_mask(os.path.join(masks_path, fname))
        save_with_gt(img, gt, pred_mask, viz_path)

        dsc = dice_score(gt, pred_mask)
        jacc = jaccard_index(gt, pred_mask)
        hd95v = hausdorff95(gt, pred_mask)

        dice_vals.append(dsc)
        jacc_vals.append(jacc)
        hd_vals.append(hd95v)

        metrics_file.write(f"{fname}: DSC={dsc:.4f}, IoU={jacc:.4f}, HD95={hd95v:.2f}\n")

    if masks_path:
        metrics_file.write("\n--- AVERAGES ---\n")
        metrics_file.write(f"Mean DSC: {np.nanmean(dice_vals):.4f}\n")
        metrics_file.write(f"Mean IoU: {np.nanmean(jacc_vals):.4f}\n")
        metrics_file.write(f"Mean HD95: {np.nanmean(hd_vals):.2f}\n")

    metrics_file.close()
    print(f"[DONE] Saved all results in: {base_dir}")


# -------------------------------------------------
# RUN DATASETS
# -------------------------------------------------
process_dataset("jsrt", "Normalized Data/jsrt/cxr", "Normalized Data/jsrt/masks")
process_dataset("synth", "Normalized Data/synth/cxr", None)
