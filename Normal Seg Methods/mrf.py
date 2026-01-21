import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import measure, morphology
from sklearn.mixture import GaussianMixture
from scipy.ndimage import binary_fill_holes
import maxflow



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
# MRF Lung Segmentation (Potts Model + Graph Cut)
# -------------------------------------------------
def mrf_lung_segmentation(img, lambda_smooth=25):
    """
    Strong Classical Pipeline:
    1. CLAHE
    2. Thoracic cavity extraction (edges + Euler)
    3. Convex hull of thoracic cavity
    4. Initial lung region from intensity valley
    5. Distance-transform filling
    6. MRF refinement inside region
    7. Final smoothing + hole removal
    """

    # --------------------------------------------------------
    # 1. CLAHE + smoothing
    # --------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    smooth = cv2.GaussianBlur(img_clahe, (7, 7), 1)
    h, w = img.shape

    # --------------------------------------------------------
    # 2. Thoracic cavity estimation (edges + fill)
    # --------------------------------------------------------
    edges = cv2.Canny(smooth, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), 2)

    cavity = binary_fill_holes(edges > 0).astype(np.uint8)

    # Remove outer background
    cavity[:30, :] = 0
    cavity[-30:, :] = 0
    cavity[:, :30] = 0
    cavity[:, -30:] = 0

    # --------------------------------------------------------
    # 3. Keep major cavity region
    # --------------------------------------------------------
    labels = measure.label(cavity, connectivity=2)
    regions = sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)
    if len(regions) > 0:
        main_cavity = np.zeros_like(cavity)
        main_cavity[labels == regions[0].label] = 1
    else:
        main_cavity = cavity

    # Convex hull for smooth cavity interior
    main_cavity = morphology.convex_hull_image(main_cavity)

    # --------------------------------------------------------
    # 4. Initial lung region via valley detection
    # --------------------------------------------------------
    # Darker than chest wall = lungs
    normalized = smooth.astype(np.float32) / 255.0
    thresh = np.percentile(normalized[main_cavity > 0], 40)
    lung_init = (normalized < thresh).astype(np.uint8)
    lung_init = lung_init * main_cavity

    # Clean
    lung_init = morphology.binary_opening(lung_init, morphology.disk(8))

    # --------------------------------------------------------
    # 5. Distance transform fill = solid lungs
    # --------------------------------------------------------
    dt = cv2.distanceTransform((lung_init > 0).astype(np.uint8), cv2.DIST_L2, 5)
    filled_lung = (dt > 5).astype(np.uint8)

    # --------------------------------------------------------
    # 6. Fit GMM **only inside this initial lung region**
    # --------------------------------------------------------
    masked_pixels = normalized[filled_lung == 1].reshape(-1, 1)
    if len(masked_pixels) < 500:
        masked_pixels = normalized.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2)
    gmm.fit(masked_pixels)

    means = gmm.means_.flatten()
    lung_label = np.argmin(means)
    bg_label = 1 - lung_label

    probs = gmm.predict_proba(normalized.reshape(-1, 1)).reshape(h, w, 2)
    data_lung = -np.log(probs[:, :, lung_label] + 1e-8)
    data_bg = -np.log(probs[:, :, bg_label] + 1e-8)

    # --------------------------------------------------------
    # 7. MRF refinement
    # --------------------------------------------------------
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((h, w))

    g.add_grid_tedges(nodes, data_lung, data_bg)
    g.add_grid_edges(nodes, lambda_smooth)
    g.maxflow()

    seg = (~g.get_grid_segments(nodes)).astype(np.uint8)

    # --------------------------------------------------------
    # 8. Restrict to cavity + clean
    # --------------------------------------------------------
    seg = seg * main_cavity

    labels = measure.label(seg, connectivity=2)
    regions = sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)

    final_mask = np.zeros_like(seg)
    for r in regions[:2]:
        final_mask[labels == r.label] = 1

    # Final smoothing
    final_mask = morphology.binary_closing(final_mask, morphology.disk(15))
    final_mask = morphology.binary_opening(final_mask, morphology.disk(10))

    return final_mask.astype(np.uint8)




def mrf_lung_segmentation(img, lambda_smooth=20):
    """
    Simplified, robust MRF pipeline.
    Good for mixed datasets (JSRT, COVIDQU, synth).
    """

    # --------------------------------------------------------
    # 1. CLAHE for global contrast correction
    # --------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enh = clahe.apply(img)

    # --------------------------------------------------------
    # 2. Otsu threshold — simple and reliable
    # --------------------------------------------------------
    _, otsu = cv2.threshold(enh, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_mask = (otsu > 0).astype(np.uint8)

    # --------------------------------------------------------
    # 3. Morphological cleaning
    # --------------------------------------------------------
    bin_mask = morphology.binary_opening(bin_mask, morphology.disk(7))
    bin_mask = morphology.binary_closing(bin_mask, morphology.disk(15))

    # --------------------------------------------------------
    # 4. Keep only the 2 largest regions (lungs)
    # --------------------------------------------------------
    labels = measure.label(bin_mask, connectivity=2)
    props = sorted(measure.regionprops(labels),
                   key=lambda r: r.area, reverse=True)

    lung_init = np.zeros_like(bin_mask)
    for r in props[:2]:
        lung_init[labels == r.label] = 1

    # --------------------------------------------------------
    # 5. Fill holes (lungs must be solid)
    # --------------------------------------------------------
    lung_init = binary_fill_holes(lung_init).astype(np.uint8)

    # --------------------------------------------------------
    # 6. Smooth silhouette
    # --------------------------------------------------------
    lung_init = morphology.binary_closing(lung_init, morphology.disk(12))

    # --------------------------------------------------------
    # 7. Fit GMM **only inside initial lung region**
    # --------------------------------------------------------
    enh_norm = enh.astype(np.float32) / 255.0

    lung_pixels = enh_norm[lung_init == 1].reshape(-1, 1)
    if len(lung_pixels) < 1000:
        lung_pixels = enh_norm.reshape(-1, 1)  # fallback

    gmm = GaussianMixture(n_components=2)
    gmm.fit(lung_pixels)

    means = gmm.means_.flatten()
    lung_label = np.argmin(means)
    bg_label = 1 - lung_label

    h, w = img.shape
    probs = gmm.predict_proba(enh_norm.reshape(-1, 1)).reshape(h, w, 2)

    data_lung = -np.log(probs[:, :, lung_label] + 1e-8)
    data_bg   = -np.log(probs[:, :, bg_label] + 1e-8)

    # --------------------------------------------------------
    # 8. MRF refinement INSIDE initial lung area
    # --------------------------------------------------------
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((h, w))

    g.add_grid_tedges(nodes, data_lung, data_bg)
    g.add_grid_edges(nodes, lambda_smooth)

    g.maxflow()
    seg = (~g.get_grid_segments(nodes)).astype(np.uint8)

    # Restrict to initial lung area
    seg = seg * lung_init

    # --------------------------------------------------------
    # 9. Final post-processing
    # --------------------------------------------------------
    labels = measure.label(seg, connectivity=2)
    props = sorted(measure.regionprops(labels),
                   key=lambda r: r.area, reverse=True)

    final_mask = np.zeros_like(seg)
    for r in props[:2]:
        final_mask[labels == r.label] = 1

    final_mask = morphology.binary_closing(final_mask, morphology.disk(10))
    final_mask = morphology.binary_opening(final_mask, morphology.disk(8))

    return final_mask.astype(np.uint8)

# -------------------------------------------------
# Ground Truth Mask Loader
# -------------------------------------------------
def read_mask(path, size=(600, 600)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


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

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
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

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------
# DATASET RUNNER
# -------------------------------------------------
def process_dataset(dataset_name, images_path, masks_path=None):
    print(f"\n=== Processing {dataset_name} Dataset ===")

    output_dir = os.path.join("outputs_MRF", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    img_files = sorted([
        f for f in os.listdir(images_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    for fname in img_files:
        img_path = os.path.join(images_path, fname)
        img = read_and_preprocess(img_path)

        pred_mask = mrf_lung_segmentation(img)

        # PATH FOR SAVING
        save_path = os.path.join(
            output_dir,
            f"{dataset_name}_LungSeg_{os.path.splitext(fname)[0]}_MRF.png"
        )

        # If GT present
        if masks_path is not None:
            mask_path = os.path.join(masks_path, fname)
            gt_mask = read_mask(mask_path)

            if gt_mask is not None:
                save_with_gt(img, gt_mask, pred_mask, save_path,
                             title=f"{dataset_name} – MRF Segmentation: {fname}")
            else:
                save_without_gt(img, pred_mask, save_path,
                                title=f"{dataset_name} – MRF Segmentation: {fname}")
        else:
            # Synth dataset – no ground truth
            save_without_gt(img, pred_mask, save_path,
                            title=f"{dataset_name} – MRF Segmentation: {fname}")

        print(f"[SAVED] {save_path}")


# -------------------------------------------------
# RUN ALL DATASETS
# -------------------------------------------------
process_dataset("COVIDQU",
                images_path="Used/COVIDQU/lung",
                masks_path="Used/COVIDQU/mask")

process_dataset("JSRT",
                images_path="Used/JSRT/lung",
                masks_path="Used/JSRT/mask")

process_dataset("synth",
                images_path="Used/synth",
                masks_path=None)
