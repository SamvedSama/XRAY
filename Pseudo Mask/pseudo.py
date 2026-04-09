import os
import cv2
import json
import numpy as np
import pandas as pd

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
from scipy.ndimage import binary_fill_holes


# =========================================================
# CONFIG
# =========================================================
TARGET_SIZE = (512, 512)

OUTPUT_DIR = "pseudo_output"
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug_images")
JSON_DIR = os.path.join(OUTPUT_DIR, "json_logs")
COMPARE_DIR = os.path.join(OUTPUT_DIR, "comparisons")

os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(COMPARE_DIR, exist_ok=True)

# Hybrid weights
ALPHA = 0.5   # pseudo dice
BETA = 0.5    # anatomical


# =========================================================
# BASIC FUNCTIONS
# =========================================================
def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot load {path}")
    mask = (mask > 127).astype(np.uint8)
    mask = cv2.resize(mask, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def dice_score(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return (2 * inter) / (a.sum() + b.sum() + 1e-8)


# =========================================================
# CLEANING + DEBUG
# =========================================================
def clean_mask(mask, debug_prefix=None):
    steps = {}

    steps["raw"] = mask.copy()

    # Remove small objects
    mask = remove_small_objects(mask.astype(bool), min_size=1000)
    mask = mask.astype(np.uint8)
    steps["small_removed"] = mask.copy()

    # Fill holes
    mask = binary_fill_holes(mask).astype(np.uint8)
    steps["holes_filled"] = mask.copy()

    # Morphology
    mask = binary_opening(mask, disk(3))
    mask = binary_closing(mask, disk(5))
    mask = mask.astype(np.uint8)
    steps["morphology"] = mask.copy()

    # Connected components
    labeled = label(mask)
    props = sorted(regionprops(labeled), key=lambda x: x.area, reverse=True)

    comp_mask = np.zeros_like(mask)
    for p in props[:2]:
        comp_mask[labeled == p.label] = 1

    steps["top2_components"] = comp_mask.copy()

    return comp_mask, steps


# =========================================================
# METRICS
# =========================================================
def compute_metrics(mask):
    metrics = {}

    total_pixels = mask.size
    lung_pixels = mask.sum()

    # Area
    area_ratio = lung_pixels / total_pixels
    metrics["area_ratio"] = float(area_ratio)

    # Components
    labeled = label(mask)
    num_components = len(regionprops(labeled))
    metrics["num_components"] = int(num_components)

    # Balance
    h, w = mask.shape
    left = mask[:, :w//2].sum()
    right = mask[:, w//2:].sum()
    balance = abs(left - right) / (left + right + 1e-8)
    metrics["balance_diff"] = float(balance)

    # Height
    ys, xs = np.where(mask > 0)
    if len(ys) > 0:
        height_ratio = (ys.max() - ys.min()) / h
    else:
        height_ratio = 0
    metrics["height_ratio"] = float(height_ratio)

    # Solidity
    solidity = 1.0 if num_components <= 2 else 0.5
    metrics["solidity"] = float(solidity)

    # Smoothness proxy
    perimeter = cv2.Canny(mask.astype(np.uint8)*255, 50, 150).sum()
    smoothness = (perimeter**2) / (lung_pixels + 1e-8)
    metrics["smoothness"] = float(smoothness)

    return metrics


# =========================================================
# ANATOMICAL SCORE
# =========================================================
def anatomical_score(metrics):
    # Simple normalized scoring (you can improve with priors)

    area = metrics["area_ratio"]
    comp = 1.0 if metrics["num_components"] == 2 else 0.5
    balance = 1 - metrics["balance_diff"]
    height = metrics["height_ratio"]
    solidity = metrics["solidity"]
    smooth = np.exp(-metrics["smoothness"] / 50)

    weights = {
        "area": 0.18,
        "comp": 0.18,
        "balance": 0.14,
        "height": 0.14,
        "solidity": 0.12,
        "smooth": 0.12
    }

    score = (
        weights["area"] * area +
        weights["comp"] * comp +
        weights["balance"] * balance +
        weights["height"] * height +
        weights["solidity"] * solidity +
        weights["smooth"] * smooth
    )

    return float(np.clip(score, 0, 1))


# =========================================================
# MAIN PROCESS FUNCTION
# =========================================================
def process_image(image_id, candidate_path, pseudo_gt_path):
    candidate = load_mask(candidate_path)
    pseudo = load_mask(pseudo_gt_path)

    # Clean candidate
    clean, steps = clean_mask(candidate)

    # Dice
    d = dice_score(clean, pseudo)

    # Metrics
    metrics = compute_metrics(clean)

    # Anatomical score
    a = anatomical_score(metrics)

    # Final score
    final_score = ALPHA * d + BETA * a

    # Save debug images
    debug_folder = os.path.join(DEBUG_DIR, image_id)
    os.makedirs(debug_folder, exist_ok=True)

    for name, img in steps.items():
        cv2.imwrite(os.path.join(debug_folder, f"{name}.png"), img*255)

    cv2.imwrite(os.path.join(debug_folder, "pseudo_gt.png"), pseudo*255)

    # Save candidate vs pseudo side-by-side comparison
    # Save candidate vs pseudo side-by-side comparison
    compare_img = np.hstack([
        (clean * 255).astype(np.uint8),
        (pseudo * 255).astype(np.uint8)
    ])

    cv2.imwrite(
        os.path.join(COMPARE_DIR, f"{image_id}_candidate_vs_pseudo.png"),
        compare_img
    )

    # JSON log
    json_data = {
        "image_id": image_id,
        "dice_pseudo": float(d),
        "anatomical_score": float(a),
        "final_score": float(final_score),
        "metrics": metrics
    }

    with open(os.path.join(JSON_DIR, f"{image_id}.json"), "w") as f:
        json.dump(json_data, f, indent=4)

    # CSV row
    row = {
        "image_id": image_id,
        "dice": d,
        "anat_score": a,
        "final_score": final_score
    }
    row.update(metrics)

    return row


# =========================================================
# BATCH PROCESSING
# =========================================================
def run_pipeline(candidate_dir, pseudo_dir, output_csv="results.csv"):
    rows = []

    files = os.listdir(candidate_dir)

    for f in files:
        candidate_path = os.path.join(candidate_dir, f)
        pseudo_path = os.path.join(pseudo_dir, f)

        if not os.path.exists(pseudo_path):
            print(f"Skipping {f}, no pseudo GT")
            continue

        image_id = os.path.splitext(f)[0]

        row = process_image(image_id, candidate_path, pseudo_path)
        rows.append(row)

        print(f"Processed {image_id} → Score: {row['final_score']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, output_csv), index=False)

    print("Done. CSV + JSON + Debug images saved.")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    run_pipeline(
        candidate_dir="../DL Methods/DeepLabV3+/outputs_single/predictions/synth",
        pseudo_dir="./UNet/outputs_unet2015_fixed/predictions/synth",
        output_csv="hybrid_results.csv"
    )