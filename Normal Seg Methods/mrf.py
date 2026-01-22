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
    """Fast HD95 using boundary sampling."""
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
# Image Reading
# -------------------------------------------------
def read_and_preprocess(path, size=(600, 600)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    return img.astype(np.uint8)


def read_mask(path, size=(600, 600)):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


# -------------------------------------------------
# MRF SEGMENTATION
# -------------------------------------------------
def mrf_lung_segmentation(img, lambda_smooth=20):

    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enh = clahe.apply(img)

    # 2. Otsu threshold
    _, otsu = cv2.threshold(enh, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_mask = (otsu > 0).astype(np.uint8)

    # 3. Morphological clean
    bin_mask = morphology.binary_opening(bin_mask, morphology.disk(7))
    bin_mask = morphology.binary_closing(bin_mask, morphology.disk(15))

    # 4. Keep largest 2 regions
    labels = measure.label(bin_mask, connectivity=2)
    props = sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)

    lung_init = np.zeros_like(bin_mask)
    for r in props[:2]:
        lung_init[labels == r.label] = 1

    # 5. Fill holes
    lung_init = binary_fill_holes(lung_init).astype(np.uint8)

    # 6. Smooth contour
    lung_init = morphology.binary_closing(lung_init, morphology.disk(12))

    # 7. Fit GMM INSIDE lung_init
    enh_norm = enh.astype(np.float32) / 255.0
    lung_pixels = enh_norm[lung_init == 1].reshape(-1, 1)

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

    # 8. Graph Cut MRF Refinement
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((h, w))
    g.add_grid_tedges(nodes, data_lung, data_bg)
    g.add_grid_edges(nodes, lambda_smooth)
    g.maxflow()

    seg = (~g.get_grid_segments(nodes)).astype(np.uint8)

    seg = seg * lung_init  # Restrict

    # 9. Final clean
    labels = measure.label(seg, connectivity=2)
    props = sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)

    final = np.zeros_like(seg)
    for r in props[:2]:
        final[labels == r.label] = 1

    final = morphology.binary_closing(final, morphology.disk(10))
    final = morphology.binary_opening(final, morphology.disk(8))

    return final.astype(np.uint8)


# -------------------------------------------------
# Visualization
# -------------------------------------------------
def save_with_gt(img, gt_mask, pred_mask, save_path, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("MRF Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_without_gt(img, pred_mask, save_path, title):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("MRF Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------------------------------------
# DATASET PROCESSOR
# -------------------------------------------------
def process_dataset(dataset_name, images_path, masks_path=None):

    print(f"\n=== Processing {dataset_name} Dataset ===")

    out_dir = os.path.join("outputs_MRF", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    metrics_file = open(os.path.join(out_dir, "metrics.txt"), "w")
    metrics_file.write(f"=== METRICS FOR {dataset_name} ===\n\n")

    dice_vals, jacc_vals, hd_vals = [], [], []

    img_files = sorted(f for f in os.listdir(images_path)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))

    for fname in img_files:
        img = read_and_preprocess(os.path.join(images_path, fname))
        pred_mask = mrf_lung_segmentation(img)

        save_path = os.path.join(out_dir, f"{dataset_name}_{fname}_MRF.png")

        if masks_path is None:
            save_without_gt(img, pred_mask, save_path, fname)
            metrics_file.write(f"{fname}: NO GT\n")
            continue

        gt = read_mask(os.path.join(masks_path, fname))

        save_with_gt(img, gt, pred_mask, save_path, fname)

        # Metrics
        dsc = dice_score(gt, pred_mask)
        jacc = jaccard_index(gt, pred_mask)
        hd95_val = hausdorff95(gt, pred_mask)

        dice_vals.append(dsc)
        jacc_vals.append(jacc)
        hd_vals.append(hd95_val)

        metrics_file.write(f"{fname}: DSC={dsc:.4f}, IoU={jacc:.4f}, HD95={hd95_val:.2f}\n")
        print(f"{fname}: DSC={dsc:.4f} | IoU={jacc:.4f} | HD95={hd95_val:.2f}")
        print(f"[SAVED] {save_path}")

    # Averages
    if masks_path:
        metrics_file.write("\n--- AVERAGES ---\n")
        metrics_file.write(f"Mean DSC: {np.nanmean(dice_vals):.4f}\n")
        metrics_file.write(f"Mean IoU: {np.nanmean(jacc_vals):.4f}\n")
        metrics_file.write(f"Mean HD95: {np.nanmean(hd_vals):.2f}\n")

    metrics_file.close()
    print(f"[METRICS SAVED] outputs_MRF/{dataset_name}/metrics.txt")


# -------------------------------------------------
# RUN ALL DATASETS
# -------------------------------------------------
process_dataset("COVIDQU", "Used/COVIDQU/lung", "Used/COVIDQU/mask")
process_dataset("JSRT",     "Used/JSRT/lung",    "Used/JSRT/mask")
process_dataset("PA",       "Used/pa/lung",      "Used/pa/mask")
process_dataset("synth",    "Used/synth",        None)
