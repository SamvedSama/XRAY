import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# PATH SETTINGS
# -----------------------------
BASE_DIR = "data/COVIDQU/Lung Segmentation Data/Test/Non-COVID"
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "lung masks")
OUT_DIR = "edge_only_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# LOAD FILENAMES
# -----------------------------
img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
mask_files = sorted([f for f in os.listdir(MASK_DIR) if f in img_files])
img_files = img_files[:10]  # process up to 10 images

# -----------------------------
# PROCESS EACH IMAGE
# -----------------------------
for fname in tqdm(img_files, desc="Performing edge-based segmentation"):
    img_path = os.path.join(IMG_DIR, fname)
    mask_path = os.path.join(MASK_DIR, fname)

    # --- Load Image and Mask ---
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or gt_mask is None:
        print(f"Skipping {fname} (missing image or mask)")
        continue

    # --- Resize for consistency ---
    img = cv2.resize(img, (1024, 1024))
    gt_mask = cv2.resize(gt_mask, (1024, 1024))

    # -------------------------------------------------
    # 1. Preprocessing: noise reduction + contrast enhancement
    # -------------------------------------------------
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # -------------------------------------------------
    # 2. Edge Detection (Canny, Sobel, Scharr)
    # -------------------------------------------------
    canny = cv2.Canny(enhanced, 30, 100)

    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))

    scharrx = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)
    scharr = cv2.convertScaleAbs(cv2.magnitude(scharrx, scharry))

    # -------------------------------------------------
    # 3. Create Visualization Panel
    # -------------------------------------------------
    def to_rgb(im):
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    img_disp = cv2.resize(to_rgb(img), (256, 256))
    gt_disp = cv2.resize(to_rgb(gt_mask), (256, 256))
    canny_disp = cv2.resize(to_rgb(canny), (256, 256))
    sobel_disp = cv2.resize(to_rgb(sobel), (256, 256))
    scharr_disp = cv2.resize(to_rgb(scharr), (256, 256))

    # Combine horizontally
    panel = np.concatenate((img_disp, gt_disp, canny_disp, sobel_disp, scharr_disp), axis=1)

    # -------------------------------------------------
    # 4. Save Output
    # -------------------------------------------------
    out_path = os.path.join(OUT_DIR, f"{os.path.splitext(fname)[0]}_edges.png")
    cv2.imwrite(out_path, panel)

print(f"\n✅ Done! Edge results saved in: {OUT_DIR}")
