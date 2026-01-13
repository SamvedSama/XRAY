import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# PATH SETTINGS
# -----------------------------
# Original dataset
BASE_DIR = "data/COVIDQU/Lung Segmentation Data/Test/Non-COVID"
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "lung masks")

# Synthetic dataset
SYNTH_BASE_DIR = "data/synth"
SYNTH_FOLDERS = [str(i) for i in range(1, 16)]  # Folders 1-15

OUT_DIR = "combined_lung_segmentation"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# LOAD FILENAMES FROM BOTH DATASETS
# -----------------------------
# Original dataset (5 images)
original_img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
original_mask_files = sorted([f for f in os.listdir(MASK_DIR) if f in original_img_files])
original_img_files = original_img_files[:5]  # process 5 images

# Synthetic dataset (10 images from random folders)
synth_img_files = []
synth_img_paths = []

# Collect all synthetic images from all folders
for folder in SYNTH_FOLDERS:
    folder_path = os.path.join(SYNTH_BASE_DIR, folder)
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for file in files:
            synth_img_paths.append(os.path.join(folder_path, file))
            synth_img_files.append(file)

# Randomly select 10 synthetic images
import random
random.seed(42)  # For reproducibility
if len(synth_img_files) > 10:
    indices = random.sample(range(len(synth_img_files)), 10)
    synth_img_files = [synth_img_files[i] for i in indices]
    synth_img_paths = [synth_img_paths[i] for i in indices]

print(f"📊 Dataset Summary:")
print(f"  Original dataset: {len(original_img_files)} images")
print(f"  Synthetic dataset: {len(synth_img_files)} images")
print(f"  Total: {len(original_img_files) + len(synth_img_files)} images")

def apply_edge_detection(image):
    """Apply edge detection methods"""
    results = {}
    results['canny'] = cv2.Canny(image, 30, 100)
    
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    results['sobel'] = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
    
    scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    results['scharr'] = cv2.convertScaleAbs(cv2.magnitude(scharrx, scharry))
    
    return results

def apply_thresholding(image):
    """Apply thresholding methods"""
    results = {}
    
    # Otsu thresholding
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['otsu'] = otsu
    
    # Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    results['adaptive'] = adaptive
    
    # Multi-Otsu thresholding
    try:
        from skimage.filters import threshold_multiotsu
        thresholds = threshold_multiotsu(image, classes=3)
        regions = np.digitize(image, bins=thresholds)
        multi_otsu = (regions == 1).astype(np.uint8) * 255
        results['multi_otsu'] = multi_otsu
    except ImportError:
        # Fallback: manual multi-thresholding
        _, binary1 = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        _, binary2 = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        multi_otsu = cv2.bitwise_and(binary1, cv2.bitwise_not(binary2))
        results['multi_otsu'] = multi_otsu
    
    return results

def post_process_edges(edge_map, original_image):
    """Enhanced edge post-processing"""
    # Dilate edges to connect gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edge_map, kernel, iterations=2)
    
    # Find and fill contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(edge_map)
    
    if contours:
        # Filter by area and aspect ratio (lungs are typically elliptical)
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area
                # Check aspect ratio (lungs are wider than tall)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratios
                    valid_contours.append(contour)
        
        cv2.drawContours(filled, valid_contours, -1, 255, -1)
    
    # Morphological closing to smooth
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
    
    return closed

def post_process_threshold(thresh_map, original_image):
    """Enhanced threshold post-processing"""
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh_map, cv2.MORPH_OPEN, kernel)
    
    # Fill holes and filter by area
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(opened)
    
    if contours:
        # Filter contours by area and shape
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Higher threshold for thresholding
                # Check circularity (lungs are somewhat circular/elliptical)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.1 < circularity < 0.8:  # Not too circular, not too irregular
                        valid_contours.append(contour)
        
        cv2.drawContours(filled, valid_contours, -1, 255, -1)
    
    return filled

def combine_edge_methods(edge_results, original_image):
    """Combine multiple edge detection methods"""
    # Method 1: Logical OR of all edge maps
    combined_or = np.zeros_like(edge_results['canny'])
    for method, edges in edge_results.items():
        combined_or = cv2.bitwise_or(combined_or, edges)
    
    # Method 2: Weighted average (normalize first)
    combined_weighted = np.zeros_like(edge_results['canny'], dtype=np.float32)
    weights = {'canny': 0.5, 'sobel': 0.3, 'scharr': 0.2}  # Canny gets highest weight
    
    for method, edges in edge_results.items():
        normalized = edges.astype(np.float32) / 255.0
        combined_weighted += normalized * weights.get(method, 0.33)
    
    combined_weighted = (combined_weighted * 255).astype(np.uint8)
    
    # Method 3: Intersection (conservative approach)
    combined_and = edge_results['canny'].copy()
    for method, edges in edge_results.items():
        if method != 'canny':  # Start with canny as base
            combined_and = cv2.bitwise_and(combined_and, edges)
    
    return {
        'edges_or': combined_or,
        'edges_weighted': combined_weighted,
        'edges_and': combined_and
    }

def combine_threshold_methods(thresh_results, original_image):
    """Combine multiple thresholding methods"""
    # Method 1: Majority voting
    thresh_arrays = list(thresh_results.values())
    stacked = np.stack(thresh_arrays, axis=0)
    majority = np.sum(stacked > 127, axis=0) > (len(thresh_arrays) // 2)
    majority_voting = (majority.astype(np.uint8) * 255)
    
    # Method 2: Weighted combination (Multi-Otsu gets highest weight)
    weights = {'otsu': 0.2, 'adaptive': 0.2, 'multi_otsu': 0.6}
    combined_weighted = np.zeros_like(thresh_results['otsu'], dtype=np.float32)
    
    for method, thresh in thresh_results.items():
        normalized = thresh.astype(np.float32) / 255.0
        combined_weighted += normalized * weights.get(method, 0.33)
    
    combined_weighted = (combined_weighted * 255).astype(np.uint8)
    
    # Method 3: Union (liberal approach)
    combined_union = thresh_results['otsu'].copy()
    for method, thresh in thresh_results.items():
        if method != 'otsu':
            combined_union = cv2.bitwise_or(combined_union, thresh)
    
    return {
        'thresh_majority': majority_voting,
        'thresh_weighted': combined_weighted,
        'thresh_union': combined_union
    }

def combine_edge_and_threshold(edge_combined, thresh_combined, original_image):
    """Combine edge and threshold results"""
    combined_results = {}
    
    for edge_name, edge_map in edge_combined.items():
        for thresh_name, thresh_map in thresh_combined.items():
            # Post-process both
            edge_processed = post_process_edges(edge_map, original_image)
            thresh_processed = post_process_threshold(thresh_map, original_image)
            
            # Method 1: Intersection (conservative)
            intersection = cv2.bitwise_and(edge_processed, thresh_processed)
            
            # Method 2: Union (liberal)
            union = cv2.bitwise_or(edge_processed, thresh_processed)
            
            # Method 3: Weighted average
            edge_weighted = edge_processed.astype(np.float32) * 0.6
            thresh_weighted = thresh_processed.astype(np.float32) * 0.4
            weighted_avg = (edge_weighted + thresh_weighted).astype(np.uint8)
            
            combined_results[f"{edge_name}_{thresh_name}_intersection"] = intersection
            combined_results[f"{edge_name}_{thresh_name}_union"] = union
            combined_results[f"{edge_name}_{thresh_name}_weighted"] = weighted_avg
    
    return combined_results

def create_comparison_panel(original, ground_truth, predictions, method_name):
    """Create comparison panel for combined methods"""
    def resize_to_display(img):
        return cv2.resize(img, (250, 250))
    
    def to_rgb(img):
        if len(img.shape) == 3:
            return img
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Resize images
    orig_disp = resize_to_display(to_rgb(original))
    gt_disp = resize_to_display(to_rgb(ground_truth))
    pred_disp = resize_to_display(to_rgb(predictions))
    
    # Create panel
    panel = np.concatenate([orig_disp, gt_disp, pred_disp], axis=1)
    
    # Add labels
    # Add labels at the bottom of each panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    
    # Add background rectangles for better text visibility
    cv2.rectangle(panel, (5, 225), (245, 245), (0, 0, 0), -1)
    cv2.rectangle(panel, (255, 225), (495, 245), (0, 0, 0), -1)
    cv2.rectangle(panel, (505, 225), (745, 245), (0, 0, 0), -1)
    
    cv2.putText(panel, "Original", (10, 240), font, font_scale, color, thickness)
    cv2.putText(panel, "Ground Truth", (260, 240), font, font_scale, color, thickness)
    cv2.putText(panel, f"{method_name}", (510, 240), font, font_scale, color, thickness)
    
    return panel

def process_single_image(img, gt_mask, output_prefix):
    """Process a single image with all combination methods"""
    # -------------------------------------------------
    # 1. Preprocessing
    # -------------------------------------------------
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # -------------------------------------------------
    # 2. Apply Individual Methods
    # -------------------------------------------------
    edge_results = apply_edge_detection(enhanced)
    thresh_results = apply_thresholding(enhanced)

    # -------------------------------------------------
    # 3. Combine Methods
    # -------------------------------------------------
    edge_combined = combine_edge_methods(edge_results, img)
    thresh_combined = combine_threshold_methods(thresh_results, img)
    all_combined = combine_edge_and_threshold(edge_combined, thresh_combined, img)

    # -------------------------------------------------
    # 4. Save Results
    # -------------------------------------------------
    base_name = output_prefix.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
    
    # Save edge combinations
    for method_name, result in edge_combined.items():
        processed = post_process_edges(result, img)
        panel = create_comparison_panel(img, gt_mask, processed, method_name)
        cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_{method_name}.png"), panel)
    
    # Save threshold combinations
    for method_name, result in thresh_combined.items():
        processed = post_process_threshold(result, img)
        panel = create_comparison_panel(img, gt_mask, processed, method_name)
        cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_{method_name}.png"), panel)
    
    # Save edge+threshold combinations (save only best ones to avoid too many files)
    best_methods = ['edges_weighted_thresh_weighted_intersection', 
                   'edges_weighted_thresh_weighted_union',
                   'edges_or_thresh_majority_weighted']
    
    for method_name in best_methods:
        if method_name in all_combined:
            panel = create_comparison_panel(img, gt_mask, all_combined[method_name], method_name)
            cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_{method_name}.png"), panel)

# -----------------------------
# PROCESS BOTH DATASETS
# -----------------------------
print("🔬 Starting Combined Lung Segmentation...")
print("Combination strategies:")
print("  Edge Combination: OR, Weighted, AND")
print("  Threshold Combination: Majority, Weighted, Union")
print("  Edge+Threshold: Intersection, Union, Weighted")
print("-" * 50)

# Process original dataset
print("\n🏥 Processing Original Dataset...")
for i, fname in enumerate(tqdm(original_img_files, desc="Original images")):
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

    # Process image
    process_single_image(img, gt_mask, f"original_{i}_{fname}")

# Process synthetic dataset
print("\n🤖 Processing Synthetic Dataset...")
for i, (fname, img_path) in enumerate(tqdm(zip(synth_img_files, synth_img_paths), desc="Synthetic images")):
    # --- Load Image (no ground truth mask for synthetic) ---
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Skipping {fname} (missing image)")
        continue

    # --- Resize for consistency ---
    img = cv2.resize(img, (1024, 1024))
    
    # Create dummy ground truth (all zeros) for display consistency
    gt_mask = np.zeros_like(img)

    # Process image
    process_single_image(img, gt_mask, f"synth_{i}_{fname}")

def process_single_image(img, gt_mask, output_prefix):
    """Process a single image with all combination methods"""
    # -------------------------------------------------
    # 1. Preprocessing
    # -------------------------------------------------
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # -------------------------------------------------
    # 2. Apply Individual Methods
    # -------------------------------------------------
    edge_results = apply_edge_detection(enhanced)
    thresh_results = apply_thresholding(enhanced)

    # -------------------------------------------------
    # 3. Combine Methods
    # -------------------------------------------------
    edge_combined = combine_edge_methods(edge_results, img)
    thresh_combined = combine_threshold_methods(thresh_results, img)
    all_combined = combine_edge_and_threshold(edge_combined, thresh_combined, img)

    # -------------------------------------------------
    # 4. Save Results
    # -------------------------------------------------
    base_name = output_prefix.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
    
    # Save edge combinations
    for method_name, result in edge_combined.items():
        processed = post_process_edges(result, img)
        panel = create_comparison_panel(img, gt_mask, processed, method_name)
        cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_{method_name}.png"), panel)
    
    # Save threshold combinations
    for method_name, result in thresh_combined.items():
        processed = post_process_threshold(result, img)
        panel = create_comparison_panel(img, gt_mask, processed, method_name)
        cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_{method_name}.png"), panel)
    
    # Save edge+threshold combinations (save only best ones to avoid too many files)
    best_methods = ['edges_weighted_thresh_weighted_intersection', 
                   'edges_weighted_thresh_weighted_union',
                   'edges_or_thresh_majority_weighted']
    
    for method_name in best_methods:
        if method_name in all_combined:
            panel = create_comparison_panel(img, gt_mask, all_combined[method_name], method_name)
            cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_{method_name}.png"), panel)

# Move the process_single_image function definition before the main processing loop

print(f"\n✅ Done! Combined results saved in: {OUT_DIR}")
print("\n🔧 Combination Strategies Explained:")
print("=" * 50)
print("EDGE COMBINATION:")
print("• OR: Union of all edge maps (liberal, catches more edges)")
print("• Weighted: Canny(0.5) + Sobel(0.3) + Scharr(0.2)")
print("• AND: Intersection (conservative, only strong edges)")
print("\nTHRESHOLD COMBINATION:")
print("• Majority: Pixel is lung if majority of methods agree")
print("• Weighted: Multi-Otsu(0.6) + Otsu(0.2) + Adaptive(0.2)")
print("• Union: Any method calling it lung counts as lung")
print("\nEDGE+THRESHOLD COMBINATION:")
print("• Intersection: Must be edge AND threshold (conservative)")
print("• Union: Edge OR threshold (liberal)")
print("• Weighted: Edges(60%) + Threshold(40%)")
print("\n🎯 Best performing combinations are saved!")
