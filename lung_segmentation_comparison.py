import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# PATH SETTINGS (same as edge_seg.py)
# -----------------------------
BASE_DIR = "data/COVIDQU/Lung Segmentation Data/Test/Non-COVID"
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "lung masks")
OUT_DIR = "lung_segmentation_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# LOAD FILENAMES (same as edge_seg.py)
# -----------------------------
img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
mask_files = sorted([f for f in os.listdir(MASK_DIR) if f in img_files])
img_files = img_files[:10]

def apply_edge_detection(image):
    """
    Apply multiple edge detection methods to segment lungs.
    
    Edge detection works by finding sharp intensity changes in the image.
    For lung X-rays, edges typically appear at the lung boundaries where
    there's a contrast between lung tissue (darker) and surrounding areas.
    """
    results = {}
    
    # Canny Edge Detection
    # Uses a two-stage algorithm: noise reduction + edge tracking via hysteresis
    # Lower threshold: minimum edge strength
    # Upper threshold: maximum edge strength
    canny = cv2.Canny(image, 30, 100)
    results['canny'] = canny
    
    # Sobel Edge Detection
    # Uses 3x3 kernels to compute gradients in X and Y directions
    # Good for detecting edges with specific orientations
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
    results['sobel'] = sobel
    
    # Scharr Edge Detection
    # Similar to Sobel but uses different kernel coefficients
    # More accurate for rotational symmetry and less sensitive to noise
    scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    scharr = cv2.convertScaleAbs(cv2.magnitude(scharrx, scharry))
    results['scharr'] = scharr
    
    return results

def apply_thresholding(image):
    """
    Apply multiple thresholding methods to segment lungs.
    
    Thresholding works by separating pixels based on intensity values.
    For lung X-rays, lung tissue typically has different intensity ranges
    compared to background, bones, and other structures.
    """
    results = {}
    
    # Otsu's Thresholding
    # Automatically determines optimal threshold by maximizing inter-class variance
    # Works well when the image has a bimodal histogram (foreground/background)
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['otsu'] = otsu
    
    # Adaptive Thresholding
    # Calculates threshold for small regions based on local neighborhood
    # Better for images with varying illumination conditions
    adaptive = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    results['adaptive'] = adaptive
    
    # Multi-Otsu Thresholding
    # Divides image into multiple classes based on intensity
    # Useful for separating lung tissue from other structures
    try:
        # Using scikit-image for multi-otsu if available
        from skimage.filters import threshold_multiotsu
        thresholds = threshold_multiotsu(image, classes=3)
        regions = np.digitize(image, bins=thresholds)
        multi_otsu = (regions == 1).astype(np.uint8) * 255  # Extract middle class
        results['multi_otsu'] = multi_otsu
    except ImportError:
        # Fallback: simple manual thresholding
        _, binary1 = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        _, binary2 = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        multi_otsu = cv2.bitwise_and(binary1, cv2.bitwise_not(binary2))
        results['multi_otsu'] = multi_otsu
    
    return results

def post_process_edges(edge_map, original_image):
    """
    Post-process edge maps to create filled lung regions.
    
    Edge detection only gives us boundaries, so we need to fill them.
    This uses morphological operations and contour filling.
    """
    # Dilate edges to connect gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edge_map, kernel, iterations=2)
    
    # Find contours and fill them
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create filled mask
    filled = np.zeros_like(edge_map)
    # Fill the largest contours (likely to be lungs)
    if contours:
        # Sort by area and keep top few largest contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        cv2.drawContours(filled, contours, -1, 255, -1)
    
    # Apply morphological closing to smooth the result
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
    
    return closed

def post_process_threshold(thresh_map, original_image):
    """
    Post-process threshold maps to clean up noise and improve lung segmentation.
    """
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh_map, cv2.MORPH_OPEN, kernel)
    
    # Fill holes
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(opened)
    
    if contours:
        # Filter contours by area to remove small noise
        min_area = 1000  # Minimum area threshold
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        cv2.drawContours(filled, valid_contours, -1, 255, -1)
    
    return filled

def create_method_comparison(original, ground_truth, predicted, method_name):
    """
    Create a simple comparison panel: Original | Ground Truth | Predicted
    """
    def resize_to_display(img):
        return cv2.resize(img, (300, 300))
    
    def to_rgb(img):
        if len(img.shape) == 3:
            return img
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Resize all images for display
    orig_disp = resize_to_display(to_rgb(original))
    gt_disp = resize_to_display(to_rgb(ground_truth))
    pred_disp = resize_to_display(to_rgb(predicted))
    
    # Create horizontal panel
    panel = np.concatenate([orig_disp, gt_disp, pred_disp], axis=1)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    
    cv2.putText(panel, "Original", (10, 30), font, font_scale, color, thickness)
    cv2.putText(panel, "Ground Truth", (310, 30), font, font_scale, color, thickness)
    cv2.putText(panel, f"Predicted ({method_name})", (610, 30), font, font_scale, color, thickness)
    
    return panel

# -----------------------------
# PROCESS EACH IMAGE
# -----------------------------
print("🔬 Starting Lung Segmentation Comparison...")
print("Methods being tested:")
print("  Edge Detection: Canny, Sobel, Scharr")
print("  Thresholding: Otsu, Adaptive, Multi-Otsu")
print("-" * 50)

for fname in tqdm(img_files, desc="Processing images"):
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
    # 2. Apply Edge Detection Methods
    # -------------------------------------------------
    edge_results = apply_edge_detection(enhanced)

    # -------------------------------------------------
    # 3. Apply Thresholding Methods
    # -------------------------------------------------
    thresh_results = apply_thresholding(enhanced)

    # -------------------------------------------------
    # 4. Create Individual Comparison Panels
    # -------------------------------------------------
    base_name = os.path.splitext(fname)[0]
    
    # Edge Detection Comparisons
    canny_pred = post_process_edges(edge_results['canny'], img)
    canny_panel = create_method_comparison(img, gt_mask, canny_pred, "Canny")
    cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_canny.png"), canny_panel)
    
    sobel_pred = post_process_edges(edge_results['sobel'], img)
    sobel_panel = create_method_comparison(img, gt_mask, sobel_pred, "Sobel")
    cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_sobel.png"), sobel_panel)
    
    scharr_pred = post_process_edges(edge_results['scharr'], img)
    scharr_panel = create_method_comparison(img, gt_mask, scharr_pred, "Scharr")
    cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_scharr.png"), scharr_panel)
    
    # Thresholding Comparisons
    otsu_pred = post_process_threshold(thresh_results['otsu'], img)
    otsu_panel = create_method_comparison(img, gt_mask, otsu_pred, "Otsu")
    cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_otsu.png"), otsu_panel)
    
    adaptive_pred = post_process_threshold(thresh_results['adaptive'], img)
    adaptive_panel = create_method_comparison(img, gt_mask, adaptive_pred, "Adaptive")
    cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_adaptive.png"), adaptive_panel)
    
    multi_otsu_pred = post_process_threshold(thresh_results['multi_otsu'], img)
    multi_otsu_panel = create_method_comparison(img, gt_mask, multi_otsu_pred, "Multi-Otsu")
    cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_multi_otsu.png"), multi_otsu_panel)

print(f"\n✅ Done! Comparison results saved in: {OUT_DIR}")
print("\n📊 Method Explanations:")
print("=" * 50)
print("EDGE DETECTION METHODS:")
print("• Canny: Uses dual thresholds and hysteresis for precise edge detection")
print("• Sobel: Computes gradients using 3x3 kernels, good for directional edges")
print("• Scharr: More accurate rotational symmetry, better for corner detection")
print("\nTHRESHOLDING METHODS:")
print("• Otsu: Automatically finds optimal threshold by maximizing class separation")
print("• Adaptive: Uses local neighborhood thresholds, handles varying illumination")
print("• Multi-Otsu: Separates into multiple classes for complex structures")
print("\nPOST-PROCESSING:")
print("• Edge maps are dilated, contoured, and filled to create solid regions")
print("• Threshold maps are filtered and holes are filled for cleaner segmentation")
