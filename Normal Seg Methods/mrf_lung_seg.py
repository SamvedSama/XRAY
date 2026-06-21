"""
Faithful MRF Lung Segmentation
Based on:
  Paper 1 - Vittitoe et al., Med. Phys. 25(6), 1998
  Paper 2 - Vittitoe et al., Med. Phys. 26(8), 1999

Architecture (exactly as in papers):
  TRAIN:
    1. Downsample training images to 64x64, quantize to 8 gray levels
    2. Build spatial prior  P_spatial[i, label]
    3. Build pairwise init  P_init[i, nb_dir, label, y_i, y_i']
    4. Build pair-clique    P_clique[i, nb_dir, label, x_i', y_i]

  TEST per image:
    1. Preprocess: 256x256 + CLAHE (user constraint) — IDENTICAL to original
    2. Downsample to 64x64, quantize to 8 gray levels
    3. Initial classification via spatial prior + pairwise gray distributions
    4. ICM: 10 iterations
    5. Upsample binary mask back to 256x256
"""

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage import measure, morphology
from skimage.morphology import disk
from scipy.ndimage import binary_dilation as ndi_dilation


def _remove_small_objects_compat(mask_bool, size_thresh):
    """
    Version-safe wrapper for skimage.morphology.remove_small_objects.
    skimage >= 0.26 renamed `min_size` -> `max_size` (objects with area
    <= max_size are removed). Older skimage only accepts `min_size`
    (objects with area < min_size are removed). Try the new kwarg first
    so this script runs unmodified on either skimage version.
    """
    try:
        return morphology.remove_small_objects(mask_bool, max_size=size_thresh)
    except TypeError:
        return morphology.remove_small_objects(mask_bool, min_size=size_thresh)


# =============================================================================
# CONSTANTS
# =============================================================================
IMG_SIZE   = (256, 256)
MRF_SIZE   = (64,  64 )
G          = 8
N_LABELS   = 2
N_DIRS     = 8
ICM_ITERS  = 10
LAPLACE    = 1.0
IMG_EXTS   = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

NEIGHBORS  = [(-1,-1),(-1,0),(-1,1),
               ( 0,-1),       (0,1),
               ( 1,-1),(1, 0),(1,1)]

DATA_ROOT  = "../Standard_Data"
SPLITS     = ["train", "val", "test"]
OUT_ROOT   = "outputs_mrf_faithful"


# =============================================================================
# PREPROCESSING  — identical to original code
# =============================================================================
def read_and_preprocess(path, size=(256, 256)):
    """Identical signature and logic to original — size param kept for compatibility."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img   = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    img   = cv2.GaussianBlur(img, (5, 5), 0)        # (5,5) — same as original
    img   = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


def read_mask(path, size=(256, 256)):
    """Identical signature and logic to original."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def find_mask_path(mask_dir, fname, split=None, category=None):
    """Identical to original."""
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
# MRF INPUT PREPARATION
# =============================================================================
def to_mrf_input(img256):
    """Downsample 256x256 to 64x64 and quantize to G=8 gray levels."""
    small = cv2.resize(img256, MRF_SIZE, interpolation=cv2.INTER_AREA)
    quant = (small.astype(np.float32) / 255.0 * (G - 1)).round().astype(np.int32)
    return np.clip(quant, 0, G - 1)


def mask_to_mrf(mask256):
    """Downsample 256x256 binary mask to 64x64."""
    small = cv2.resize(mask256, MRF_SIZE, interpolation=cv2.INTER_NEAREST)
    return (small > 0).astype(np.int32)


# =============================================================================
# METRIC FUNCTIONS  — identical names, identical math to original
# =============================================================================
BORDER_FOOTPRINT = disk(1)

def _get_border(mask):
    """1-pixel border via dilation XOR original."""
    mask_bool = mask.astype(bool)
    return ndi_dilation(mask_bool, structure=BORDER_FOOTPRINT) ^ mask_bool


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
# TRAINING — build MRF lookup tables from labeled images
# =============================================================================
def train_mrf(img_mask_pairs):
    """
    Build three lookup tables from (img256, mask256) pairs.
    Paper: Section III.B-D, Vittitoe et al. 1998.

    Tables:
      log_P_spatial[H, W, N_LABELS]
          log P(x_i=label) at each pixel — spatial prior.

      log_P_init[H, W, N_DIRS, N_LABELS, G, G]
          log P(y_i, y_i' | x_i=label) per pixel per direction.
          Used in initial classification (paper eq 24-25).

      log_P_clique[H, W, N_DIRS, N_LABELS, N_LABELS, G]
          log P(x_i=label | x_i'=xi', y_i=yi) per pixel per direction.
          Used as ICM pair-clique potential (paper eq 2, Paper 2).
    """
    H, W = MRF_SIZE

    count_spatial = np.zeros((H, W, N_LABELS),                    dtype=np.float64)
    count_init    = np.zeros((H, W, N_DIRS, N_LABELS, G, G),      dtype=np.float64)
    count_clique  = np.zeros((H, W, N_DIRS, N_LABELS, N_LABELS, G), dtype=np.float64)

    n_train = len(img_mask_pairs)
    print(f"    Training on {n_train} image-mask pairs...")

    for idx, (img256, mask256) in enumerate(img_mask_pairs):
        Y = to_mrf_input(img256)       # (64,64) int [0, G-1]
        X = mask_to_mrf(mask256)       # (64,64) int {0,1}

        # Spatial prior counts
        for label in range(N_LABELS):
            count_spatial[:, :, label] += (X == label).astype(np.float64)

        # Pairwise counts over all 8 directions
        for d, (dr, dc) in enumerate(NEIGHBORS):
            r0, r1 = max(0, -dr), min(H, H - dr)
            c0, c1 = max(0, -dc), min(W, W - dc)

            yi  = Y[r0:r1, c0:c1]
            xi  = X[r0:r1, c0:c1]
            yi_ = Y[r0+dr:r1+dr, c0+dc:c1+dc]
            xi_ = X[r0+dr:r1+dr, c0+dc:c1+dc]

            rows = (np.arange(r0, r1)[:, None]
                    * np.ones((1, c1 - c0), dtype=int))
            cols = (np.ones((r1 - r0, 1), dtype=int)
                    * np.arange(c0, c1)[None, :])

            # P(y_i, y_i' | x_i) for initial classification
            np.add.at(count_init,   (rows, cols, d, xi,  yi, yi_), 1.0)

            # Joint (x_i, x_i', y_i) for clique potentials
            np.add.at(count_clique, (rows, cols, d, xi, xi_, yi),  1.0)

        if (idx + 1) % 10 == 0:
            print(f"      processed {idx + 1}/{n_train}")

    # ── Laplace smoothing + log conversion ───────────────────────────────────

    # 1. Spatial prior
    count_spatial += LAPLACE
    total_spatial  = count_spatial.sum(axis=2, keepdims=True)
    log_P_spatial  = np.log(count_spatial / total_spatial)

    # 2. Init distributions: normalize over gray-level pairs (axes 4,5)
    count_init += LAPLACE
    norm_init   = count_init.sum(axis=(4, 5), keepdims=True)
    log_P_init  = np.log(count_init / (norm_init + 1e-30))

    # 3. Clique potentials: normalize over x_i (axis 3)
    count_clique += LAPLACE
    norm_clique   = count_clique.sum(axis=3, keepdims=True)
    log_P_clique  = np.log(count_clique / (norm_clique + 1e-30))

    print(f"    Training complete.")
    return {
        "log_P_spatial": log_P_spatial,
        "log_P_init":    log_P_init,
        "log_P_clique":  log_P_clique,
    }


# =============================================================================
# INITIAL CLASSIFICATION  (paper Section III.D, eq 22-25)
# =============================================================================
def initial_classify(Y, tables):
    """
    For each pixel i, choose label maximizing:
        P(x_i | y_i, y_Ni) ∝ P(y_i, y_Ni | x_i) · P(x_i)
    P(y_i, y_Ni | x_i) approximated as product of pair terms (eq 24).
    """
    H, W          = MRF_SIZE
    log_P_spatial = tables["log_P_spatial"]
    log_P_init    = tables["log_P_init"]

    score = log_P_spatial.copy()   # (H, W, N_LABELS)

    for d, (dr, dc) in enumerate(NEIGHBORS):
        r0, r1 = max(0, -dr), min(H, H - dr)
        c0, c1 = max(0, -dc), min(W, W - dc)

        yi  = Y[r0:r1, c0:c1]
        yi_ = Y[r0+dr:r1+dr, c0+dc:c1+dc]

        rows = (np.arange(r0, r1)[:, None]
                * np.ones((1, c1 - c0), dtype=int))
        cols = (np.ones((r1 - r0, 1), dtype=int)
                * np.arange(c0, c1)[None, :])

        for label in range(N_LABELS):
            score[r0:r1, c0:c1, label] += \
                log_P_init[rows, cols, d, label, yi, yi_]

    return np.argmax(score, axis=2).astype(np.int32)


# =============================================================================
# ICM SEGMENTATION  (Besag 1986, paper Section III.C)
# =============================================================================
def icm_segment(Y, x_init, tables, n_iters=ICM_ITERS):
    """
    Iteratively update each pixel's label to maximize:
        sum_d  log P(x_i | x_i'_d, y_i)
    using current neighbor labels x_Ni (updated in-place each iteration).
    """
    H, W         = MRF_SIZE
    log_P_clique = tables["log_P_clique"]   # (H,W,N_DIRS,N_LABELS,N_LABELS,G)

    X = x_init.copy()

    for iteration in range(n_iters):
        score = np.zeros((H, W, N_LABELS), dtype=np.float64)

        for d, (dr, dc) in enumerate(NEIGHBORS):
            r0, r1 = max(0, -dr), min(H, H - dr)
            c0, c1 = max(0, -dc), min(W, W - dc)

            yi  = Y[r0:r1, c0:c1]
            xi_ = X[r0+dr:r1+dr, c0+dc:c1+dc]

            rows = (np.arange(r0, r1)[:, None]
                    * np.ones((1, c1 - c0), dtype=int))
            cols = (np.ones((r1 - r0, 1), dtype=int)
                    * np.arange(c0, c1)[None, :])

            for label in range(N_LABELS):
                score[r0:r1, c0:c1, label] += \
                    log_P_clique[rows, cols, d, label, xi_, yi]

        X_new   = np.argmax(score, axis=2).astype(np.int32)
        changed = np.sum(X_new != X)
        X       = X_new

        if changed == 0:
            print(f"      ICM converged at iteration {iteration + 1}")
            break

    return X


# =============================================================================
# POST-PROCESSING  — minimal, MRF does the heavy lifting
# =============================================================================
def postprocess(seg64, target_size=(256, 256)):
    """
    Fill holes, keep top-2 blobs, upsample to 256x256.
    No heavy morphology — the MRF handles spatial coherence.
    """
    seg = binary_fill_holes(seg64.astype(bool)).astype(np.uint8)
    seg = _remove_small_objects_compat(
          seg.astype(bool), 50).astype(np.uint8)
    labels = measure.label(seg, connectivity=2)
    props  = sorted(measure.regionprops(labels),
                    key=lambda r: r.area, reverse=True)
    clean  = np.zeros_like(seg)
    for r in props[:2]:
        clean[labels == r.label] = 1

    clean = binary_fill_holes(clean).astype(np.uint8)
    out   = cv2.resize(clean, target_size,
                       interpolation=cv2.INTER_NEAREST)
    return out.astype(np.uint8)


# =============================================================================
# VISUALIZATION  — identical structure and subplot titles to original
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
    plt.imshow(pred, cmap="gray"); plt.title("MRF Prediction"); plt.axis("off")

    if title:
        plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# =============================================================================
# DATA COLLECTION
# =============================================================================
def collect_images_and_masks(split, category):
    """Collect all (img256, mask256, fname) tuples for a split/category."""
    img_dir  = os.path.join(DATA_ROOT, split, category, "cxr")
    mask_dir = os.path.join(DATA_ROOT, split, category, "masks")

    if not os.path.isdir(img_dir):
        return [], []

    has_gt = os.path.isdir(mask_dir)
    files  = sorted(f for f in os.listdir(img_dir)
                    if os.path.splitext(f)[1].lower() in IMG_EXTS)

    with_gt    = []
    without_gt = []

    for fname in files:
        try:
            img = read_and_preprocess(os.path.join(img_dir, fname),
                                      size=IMG_SIZE)
        except Exception as e:
            print(f"    [ERR loading] {fname}: {e}")
            continue

        if has_gt:
            mp = find_mask_path(mask_dir, fname,
                                split=split, category=category)
            if mp is not None:
                try:
                    mask = read_mask(mp, size=IMG_SIZE)
                    with_gt.append((img, mask, fname))
                    continue
                except Exception as e:
                    print(f"    [ERR mask] {fname}: {e}")

        without_gt.append((img, None, fname))

    return with_gt, without_gt


# =============================================================================
# SINGLE SPLIT + CATEGORY PROCESSOR
# =============================================================================
def process_split_category(split, category, all_splits_data):
    img_dir = os.path.join(DATA_ROOT, split, category, "cxr")
    if not os.path.isdir(img_dir):
        return None

    tag         = f"{split}/{category}"
    base_out    = os.path.join(OUT_ROOT, split, category)
    masks_out   = os.path.join(base_out, "pred_masks")
    results_out = os.path.join(base_out, "comparisons")
    os.makedirs(masks_out,   exist_ok=True)
    os.makedirs(results_out, exist_ok=True)

    print(f"\n  --- {tag} ---")

    with_gt, without_gt = all_splits_data[(split, category)]
    all_items = with_gt + without_gt

    if not all_items:
        print(f"  [SKIP] no images found")
        return None

    # Build training pool from OTHER splits/categories
    train_pairs = []
    for (sp, cat), (wgt, _) in all_splits_data.items():
        if sp == split and cat == category:
            continue
        train_pairs.extend([(img, mask) for img, mask, _ in wgt])

    if len(train_pairs) < 5:
        print(f"    [WARN] No external training data — using own GT pairs (80%)")
        if len(with_gt) < 2:
            print(f"    [SKIP] Not enough GT to train MRF")
            return None
        n_train    = max(1, int(0.8 * len(with_gt)))
        train_pairs = [(img, mask) for img, mask, _ in with_gt[:n_train]]

    # Train
    tables = train_mrf(train_pairs)

    # Inference + metrics — identical file format to original
    accum        = {k: [] for k in ["DSC", "IoU", "Sens", "Spec",
                                     "MCC", "Kappa", "HD95", "NSD"]}
    metrics_path = os.path.join(base_out, "metrics.txt")

    with open(metrics_path, "w") as mf:
        mf.write(f"=== METRICS: {tag} ===\n\n")

        for (img, mask, fname) in all_items:
            try:
                Y       = to_mrf_input(img)
                x_init  = initial_classify(Y, tables)
                x_opt   = icm_segment(Y, x_init, tables, n_iters=ICM_ITERS)
                pred    = postprocess(x_opt, target_size=IMG_SIZE)

                stem       = os.path.splitext(fname)[0]
                pred_fname = stem + ".png"
                plt.imsave(os.path.join(masks_out, pred_fname),
                           pred, cmap="gray")

                gt = None
                if mask is not None:
                    gt = mask
                    m  = compute_all_metrics(gt, pred)
                    for k in accum:
                        accum[k].append(m[k])
                    # Identical format to original
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
                    path=os.path.join(results_out, f"{stem}_MRF.png"),
                    gt=gt, title=f"{tag} | {fname}"
                )
                print(f"    [OK] {fname}")

            except Exception as e:
                print(f"    [ERR] {fname}: {e}")
                with open(metrics_path, "a") as mf2:
                    mf2.write(f"{fname}: ERROR — {e}\n")

        # Identical average format to original
        if any(accum["DSC"]):
            mf.write("\n--- AVERAGE METRICS ---\n")
            labels_txt = ["DSC  ", "IoU  ", "Sens ", "Spec ",
                          "MCC  ", "Kappa", "HD95 ", "NSD  "]
            keys       = ["DSC",   "IoU",   "Sens",  "Spec",
                          "MCC",   "Kappa", "HD95",  "NSD" ]
            for label, key in zip(labels_txt, keys):
                vals = accum[key]
                if vals:
                    mf.write(
                        f"  Mean {label}: {np.nanmean(vals):.4f} "
                        f"(n={len(vals)})\n"
                    )

    print(f"    => outputs saved: {base_out}/")
    return accum if any(accum["DSC"]) else None


# =============================================================================
# GLOBAL SUMMARY  — identical format to original
# =============================================================================
def write_global_summary(all_results):
    summary_path = os.path.join(OUT_ROOT, "GLOBAL_SUMMARY.txt")
    os.makedirs(OUT_ROOT, exist_ok=True)

    global_accum = {k: [] for k in ["DSC", "IoU", "Sens", "Spec",
                                     "MCC", "Kappa", "HD95", "NSD"]}

    with open(summary_path, "w") as sf:
        sf.write("=" * 60 + "\n")
        sf.write("  GLOBAL SUMMARY — MRF Lung Segmentation\n")
        sf.write("=" * 60 + "\n\n")

        for (split, cat), accum in sorted(all_results.items()):
            if not accum:
                sf.write(f"  {split}/{cat}: no GT masks\n")
                continue
            n = len(accum.get("DSC", []))
            if n == 0:
                sf.write(f"  {split}/{cat}: 0 evaluated images\n")
                continue

            sf.write(f"\n  [{split}/{cat}]  (n={n})\n")
            for key in ["DSC", "IoU", "Sens", "Spec",
                        "MCC", "Kappa", "HD95", "NSD"]:
                vals = accum[key]
                sf.write(f"    {key:6s}: {np.nanmean(vals):.4f}\n")
                global_accum[key].extend(vals)

        sf.write("\n" + "=" * 60 + "\n")
        sf.write("  OVERALL MEAN (all categories with GT)\n")
        sf.write("=" * 60 + "\n")
        for key in ["DSC", "IoU", "Sens", "Spec",
                    "MCC", "Kappa", "HD95", "NSD"]:
            vals = global_accum[key]
            if vals:
                sf.write(
                    f"  {key:6s}: {np.nanmean(vals):.4f}  "
                    f"(n={len(vals)})\n"
                )

    print(f"\n{'='*60}")
    print(f"  Global summary written → {summary_path}")
    print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  Lung Segmentation — Faithful MRF (Vittitoe 1998/1999)")
    print("  MRF: 64×64, 8 gray levels, ICM 10 iters")
    print("  I/O: 256×256 with CLAHE")
    print("=" * 60)

    # Pass 1: collect all data
    print("\n[Pass 1] Collecting all images and masks...")
    all_splits_data = {}
    for split in SPLITS:
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.isdir(split_dir):
            print(f"  [SKIP] {split} not found")
            continue
        for category in sorted(os.listdir(split_dir)):
            if not os.path.isdir(os.path.join(split_dir, category)):
                continue
            wgt, wogt = collect_images_and_masks(split, category)
            all_splits_data[(split, category)] = (wgt, wogt)
            print(f"  {split}/{category}: "
                  f"{len(wgt)} with GT, {len(wogt)} without GT")

    # Pass 2: train + infer per split/category
    print("\n[Pass 2] Training MRF and running inference...")
    all_results = {}
    for (split, category) in all_splits_data:
        accum = process_split_category(split, category, all_splits_data)
        all_results[(split, category)] = accum

    write_global_summary(all_results)


if __name__ == "__main__":
    main()
