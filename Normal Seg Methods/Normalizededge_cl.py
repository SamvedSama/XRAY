import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.ndimage import binary_dilation as ndi_dilation
from skimage.morphology import disk

BORDER_FOOTPRINT = disk(1)

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
# Root of the standardized dataset
DATA_ROOT = "../Standard_Data"

# For IP / edge-detection methods: use ALL images from every split and category.
# Each entry is (split, category). The script auto-discovers what exists on disk.
SPLITS      = ["train", "val", "test"]

# Categories that live under each split (some may be absent in certain splits —
# the script skips silently if the folder doesn't exist).
CATEGORIES  = ["covid-19", "covid19", "indiana", "jsrt", "normal",
               "pneumonia", "tb", "synth"]

# Output root
OUT_ROOT    = "outputs_normal_edge"

# Supported image extensions (pneumonia uses .bmp)
IMG_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _get_border(mask):
    """1-pixel border via dilation XOR original."""
    mask_bool = mask.astype(bool)
    return ndi_dilation(mask_bool, structure=BORDER_FOOTPRINT) ^ mask_bool

# =============================================================================
# METRIC FUNCTIONS
# =============================================================================
def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    return (2.0 * intersection) / (np.sum(gt) + np.sum(pred) + 1e-8)


def jaccard_index(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    return intersection / (union + 1e-8)


def sensitivity(gt, pred):
    tp = np.sum(gt * pred)
    fn = np.sum(gt * (1 - pred))
    return tp / (tp + fn + 1e-8)


def specificity(gt, pred):
    tn = np.sum((1 - gt) * (1 - pred))
    fp = np.sum((1 - gt) * pred)
    return tn / (tn + fp + 1e-8)


def matthews_correlation_coefficient(gt, pred):
    tp = float(np.sum(gt * pred))
    tn = float(np.sum((1 - gt) * (1 - pred)))
    fp = float(np.sum((1 - gt) * pred))
    fn = float(np.sum(gt * (1 - pred)))
    num = (tp * tn) - (fp * fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-8
    return num / den


def cohens_kappa(gt, pred):
    tp = np.sum(gt * pred)
    tn = np.sum((1 - gt) * (1 - pred))
    fp = np.sum((1 - gt) * pred)
    fn = np.sum(gt * (1 - pred))
    n        = tp + tn + fp + fn
    po       = (tp + tn) / (n + 1e-8)
    p_gt_pos = (tp + fn) / (n + 1e-8)
    p_gt_neg = (tn + fp) / (n + 1e-8)
    p_pr_pos = (tp + fp) / (n + 1e-8)
    p_pr_neg = (tn + fn) / (n + 1e-8)
    pe       = (p_gt_pos * p_pr_pos) + (p_gt_neg * p_pr_neg)
    return (po - pe) / (1.0 - pe + 1e-8)


def hausdorff95(gt, pred):
    gt_border   = _get_border(np.asarray(gt, dtype=bool))
    pred_border = _get_border(np.asarray(pred, dtype=bool))
    if not gt_border.any() or not pred_border.any():
        return np.nan
    d_gt_to_pred = distance_transform_edt(~pred_border)[gt_border]
    d_pred_to_gt = distance_transform_edt(~gt_border)[pred_border]
    return float(np.percentile(np.hstack([d_gt_to_pred, d_pred_to_gt]), 95))


def normalized_surface_distance(gt, pred, delta=2.0):
    gt_border   = _get_border(np.asarray(gt, dtype=bool))
    pred_border = _get_border(np.asarray(pred, dtype=bool))
    if not gt_border.any() or not pred_border.any():
        return np.nan
    d_gt_to_pred = distance_transform_edt(~pred_border)[gt_border]
    d_pred_to_gt = distance_transform_edt(~gt_border)[pred_border]
    gt_within   = np.sum(d_gt_to_pred <= delta)
    pred_within = np.sum(d_pred_to_gt <= delta)
    return (gt_within + pred_within) / (len(d_gt_to_pred) + len(d_pred_to_gt) + 1e-8)


def compute_all_metrics(gt, pred):
    return {
        "DSC":   dice_score(gt, pred),
        "IoU":   jaccard_index(gt, pred),
        "Sens":  sensitivity(gt, pred),
        "Spec":  specificity(gt, pred),
        "MCC":   matthews_correlation_coefficient(gt, pred),
        "Kappa": cohens_kappa(gt, pred),
        "HD95":  hausdorff95(gt, pred),
        "NSD":   normalized_surface_distance(gt, pred, delta=2.0),
    }


# =============================================================================
# PREPROCESSING
# =============================================================================
def read_and_preprocess(path, size=(600, 600)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img   = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    img   = cv2.GaussianBlur(img, (5, 5), 0)
    img   = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


# =============================================================================
# EULER-BASED EXTRACTION
# =============================================================================
def euler_binary_extraction(img):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best = np.zeros_like(img, dtype=np.uint8)
    for candidate in [binary, cv2.bitwise_not(binary)]:
        b      = (candidate > 0).astype(np.uint8)
        labels = measure.label(b, connectivity=2)
        props  = measure.regionprops(labels)
        cleaned = np.zeros_like(b)
        for r in props:
            if r.area > 5000 and r.euler_number <= 0:
                cleaned[labels == r.label] = 1
        if cleaned.sum() > best.sum():
            best = cleaned
    best = morphology.closing(best, morphology.disk(5))
    best = morphology.opening(best, morphology.disk(3))
    return best.astype(np.uint8)


# =============================================================================
# EDGE EXTRACTION + MORPHOLOGY
# =============================================================================
def canny_edge(binary):
    return cv2.Canny(binary * 255, 50, 150)


def morphological_refinement(e):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    d = cv2.dilate(e, kernel, iterations=2)
    e = cv2.erode(d,  kernel, iterations=2)
    return e


# =============================================================================
# HOLE FILLING + LUNG EXTRACTION
# =============================================================================
def fill_lung_regions(edge_img):
    return binary_fill_holes(edge_img > 0).astype(np.uint8)


def extract_lungs(filled):
    labels = measure.label(filled, connectivity=2)
    props  = sorted(measure.regionprops(labels), key=lambda x: x.area, reverse=True)
    lung_mask = np.zeros_like(filled)
    for r in props[:2]:
        lung_mask[labels == r.label] = 1
    lung_mask = morphology.closing(lung_mask, morphology.disk(10))
    lung_mask = morphology.opening(lung_mask, morphology.disk(5))
    lung_mask = binary_fill_holes(lung_mask).astype(np.uint8)
    return lung_mask


# =============================================================================
# MAIN SEGMENTATION PIPELINE
# =============================================================================
def lung_segmentation_pipeline(img):
    binary    = euler_binary_extraction(img)
    edges     = canny_edge(binary)
    morph     = morphological_refinement(edges)
    filled    = fill_lung_regions(morph)
    lung_mask = extract_lungs(filled)
    return lung_mask


# =============================================================================
# MASK READER
# =============================================================================
def read_mask(path, size=(600, 600)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


import re

def find_mask_path(mask_dir, fname, split=None, category=None):
    """
    Try exact filename match first, then swap extension for cross-format cases
    (e.g. pneumonia .bmp CXR vs .png mask).

    Special case: val/pneumonia uses CXR_Images_<id>.bmp <-> CXR_Mask_<id>.bmp
    naming, matched by trailing numeric ID.
    """
    exact = os.path.join(mask_dir, fname)
    if os.path.exists(exact):
        return exact

    stem = os.path.splitext(fname)[0]
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        candidate = os.path.join(mask_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate

    if split == "val" and category == "pneumonia":
        match = re.search(r'(\d+)$', stem)
        if match:
            num = match.group(1)
            for mf in os.listdir(mask_dir):
                mstem = os.path.splitext(mf)[0]
                m2 = re.search(r'(\d+)$', mstem)
                if m2 and m2.group(1) == num:
                    return os.path.join(mask_dir, mf)

    return None


# =============================================================================
# VISUALIZATION
# =============================================================================
def save_comparison(img, pred, path, gt=None, title=""):
    n_cols = 3 if gt is not None else 2
    plt.figure(figsize=(4 * n_cols, 4))

    plt.subplot(1, n_cols, 1)
    plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")

    if gt is not None:
        plt.subplot(1, n_cols, 2)
        plt.imshow(gt, cmap="gray"); plt.title("Ground Truth"); plt.axis("off")

    plt.subplot(1, n_cols, n_cols)
    plt.imshow(pred, cmap="gray"); plt.title("Predicted"); plt.axis("off")

    if title:
        plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)   # 150 dpi keeps file sizes manageable
    plt.close()


# =============================================================================
# SINGLE SPLIT+CATEGORY PROCESSOR
# =============================================================================
def process_split_category(split, category):
    """
    Process all images in Standard_Data/<split>/<category>/cxr/.
    GT masks are read from Standard_Data/<split>/<category>/masks/ if present.
    Outputs go to outputs_normal_edge/<split>/<category>/.
    """
    img_dir  = os.path.join(DATA_ROOT, split, category, "cxr")
    mask_dir = os.path.join(DATA_ROOT, split, category, "masks")

    # Skip if the image directory doesn't exist for this split/category combo
    if not os.path.isdir(img_dir):
        return

    has_gt = os.path.isdir(mask_dir)

    tag        = f"{split}/{category}"
    base_out   = os.path.join(OUT_ROOT, split, category)
    masks_out  = os.path.join(base_out, "pred_masks")
    results_out = os.path.join(base_out, "comparisons")
    os.makedirs(masks_out,   exist_ok=True)
    os.makedirs(results_out, exist_ok=True)

    files = sorted(
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )

    if not files:
        print(f"  [SKIP] {tag} — no images found")
        return

    print(f"\n  --- {tag} ({len(files)} images, GT={'yes' if has_gt else 'no'}) ---")

    # Per-metric accumulators for this split/category
    accum = {k: [] for k in ["DSC", "IoU", "Sens", "Spec", "MCC", "Kappa", "HD95", "NSD"]}

    metrics_path = os.path.join(base_out, "metrics.txt")
    with open(metrics_path, "w") as mf:
        mf.write(f"=== METRICS: {tag} ===\n\n")

        for fname in files:
            try:
                img_path = os.path.join(img_dir, fname)
                img      = read_and_preprocess(img_path)
                pred     = lung_segmentation_pipeline(img)

                # Save predicted mask (always as PNG for consistency)
                stem       = os.path.splitext(fname)[0]
                pred_fname = stem + ".png"
                plt.imsave(os.path.join(masks_out, pred_fname), pred, cmap="gray")

                gt = None
                if has_gt:
                    mask_path = find_mask_path(mask_dir, fname, split=split, category=category)
                    if mask_path is None:
                        mf.write(f"{fname}: GT mask not found — skipped metrics\n")
                        print(f"    [WARN] {fname} — GT mask missing")
                    else:
                        gt = read_mask(mask_path)
                        m  = compute_all_metrics(gt, pred)
                        for k in accum:
                            accum[k].append(m[k])
                        mf.write(
                            f"{fname}: DSC={m['DSC']:.4f}, IoU={m['IoU']:.4f}, "
                            f"Sens={m['Sens']:.4f}, Spec={m['Spec']:.4f}, "
                            f"MCC={m['MCC']:.4f}, Kappa={m['Kappa']:.4f}, "
                            f"HD95={m['HD95']:.2f}, NSD={m['NSD']:.4f}\n"
                        )
                else:
                    mf.write(f"{fname}: pred saved (no GT)\n")

                save_comparison(
                    img, pred,
                    path=os.path.join(results_out, f"{stem}_compare.png"),
                    gt=gt, title=f"{tag} | {fname}"
                )
                print(f"    [OK] {fname}")

            except Exception as e:
                print(f"    [ERR] {fname}: {e}")
                with open(metrics_path, "a") as mf2:
                    mf2.write(f"{fname}: ERROR — {e}\n")

        # Write averages
        if has_gt and any(accum["DSC"]):
            mf.write("\n--- AVERAGE METRICS ---\n")
            labels = ["DSC  ", "IoU  ", "Sens ", "Spec ", "MCC  ", "Kappa", "HD95 ", "NSD  "]
            keys   = ["DSC",   "IoU",   "Sens",  "Spec",  "MCC",   "Kappa", "HD95",  "NSD" ]
            for label, key in zip(labels, keys):
                vals = accum[key]
                if vals:
                    mf.write(f"  Mean {label}: {np.nanmean(vals):.4f} "
                             f"(n={len(vals)})\n")

    print(f"    => outputs saved: {base_out}/")
    return accum if has_gt else None


# =============================================================================
# GLOBAL SUMMARY WRITER
# =============================================================================
def write_global_summary(all_results):
    """
    Writes a single summary file aggregating mean metrics across all
    split/category combos that had GT masks.
    """
    summary_path = os.path.join(OUT_ROOT, "GLOBAL_SUMMARY.txt")
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Aggregate across everything
    global_accum = {k: [] for k in ["DSC", "IoU", "Sens", "Spec", "MCC", "Kappa", "HD95", "NSD"]}

    with open(summary_path, "w") as sf:
        sf.write("=" * 60 + "\n")
        sf.write("  GLOBAL SUMMARY — Edge Detection Lung Segmentation\n")
        sf.write("=" * 60 + "\n\n")

        for (split, cat), accum in sorted(all_results.items()):
            if accum is None:
                sf.write(f"  {split}/{cat}: no GT masks\n")
                continue
            n = len(accum.get("DSC", []))
            if n == 0:
                sf.write(f"  {split}/{cat}: 0 evaluated images\n")
                continue

            sf.write(f"\n  [{split}/{cat}]  (n={n})\n")
            for key in ["DSC", "IoU", "Sens", "Spec", "MCC", "Kappa", "HD95", "NSD"]:
                vals = accum[key]
                sf.write(f"    {key:6s}: {np.nanmean(vals):.4f}\n")
                global_accum[key].extend(vals)

        sf.write("\n" + "=" * 60 + "\n")
        sf.write("  OVERALL MEAN (all categories with GT)\n")
        sf.write("=" * 60 + "\n")
        for key in ["DSC", "IoU", "Sens", "Spec", "MCC", "Kappa", "HD95", "NSD"]:
            vals = global_accum[key]
            if vals:
                sf.write(f"  {key:6s}: {np.nanmean(vals):.4f}  (n={len(vals)})\n")

    print(f"\n{'='*60}")
    print(f"  Global summary written → {summary_path}")
    print(f"{'='*60}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    print("=" * 60)
    print("  Lung Segmentation — Edge Detection (Standard_Data)")
    print("  Mode: ALL splits × ALL categories")
    print("=" * 60)

    all_results = {}   # (split, category) -> accum dict or None

    for split in SPLITS:
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.isdir(split_dir):
            print(f"\n[SKIP] Split '{split}' not found at {split_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  SPLIT: {split}")
        print(f"{'='*60}")

        # Discover categories that actually exist under this split
        present_categories = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )

        if not present_categories:
            print(f"  No category subdirectories found under {split_dir}")
            continue

        for category in present_categories:
            accum = process_split_category(split, category)
            all_results[(split, category)] = accum

    write_global_summary(all_results)


if __name__ == "__main__":
    main()