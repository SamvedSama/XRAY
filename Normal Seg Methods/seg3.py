import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes


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
# Euler-number–based Binary Extraction
# -------------------------------------------------
def euler_binary_extraction(img):
    """
    Foreground extraction preserving large connected
    components with holes (lung anatomy).
    """

    # Otsu binarization
    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary = (binary > 0).astype(np.uint8)

    labels = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labels)

    cleaned = np.zeros_like(binary)

    for region in regions:
        # Lung regions are large and contain holes
        if region.area > 5000 and region.euler_number <= 0:
            cleaned[labels == region.label] = 1

    # Smooth binary regions
    cleaned = morphology.binary_closing(cleaned, morphology.disk(5))
    cleaned = morphology.binary_opening(cleaned, morphology.disk(3))

    return cleaned.astype(np.uint8)


# -------------------------------------------------
# Edge Detection & Morphology
# -------------------------------------------------
# -------------------------------------------------
# Sobel Edge Detection
# -------------------------------------------------
def sobel_edge(binary_img):
    # Sobel gradients
    gx = cv2.Sobel(binary_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(binary_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    # Gradient magnitude
    mag = cv2.magnitude(gx, gy)

    # Normalize
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype(np.uint8)

    # Binary threshold to get edges
    _, edges = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)

    return edges



def morphological_refinement(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded


# -------------------------------------------------
# Region Filling & Lung Extraction
# -------------------------------------------------
def fill_lung_regions(edge_img):
    filled = binary_fill_holes(edge_img > 0)
    return filled.astype(np.uint8)


def extract_lungs(filled_img):
    labels = measure.label(filled_img, connectivity=2)
    regions = measure.regionprops(labels)

    regions = sorted(regions, key=lambda r: r.area, reverse=True)

    lung_mask = np.zeros_like(filled_img)

    for region in regions[:2]:  # Left & right lung
        lung_mask[labels == region.label] = 1

    lung_mask = morphology.binary_closing(lung_mask, morphology.disk(10))
    lung_mask = morphology.binary_opening(lung_mask, morphology.disk(5))

    return lung_mask.astype(np.uint8)


# -------------------------------------------------
# Segmentation Pipeline
# -------------------------------------------------
def lung_segmentation_pipeline_from_image(img):
    binary = euler_binary_extraction(img)
    edges = sobel_edge(binary)
    refined_edges = morphological_refinement(edges)
    filled = fill_lung_regions(refined_edges)
    lung_mask = extract_lungs(filled)

    segmented = img * lung_mask
    return lung_mask, segmented



# -------------------------------------------------
# Ground Truth Mask Loader
# -------------------------------------------------
def read_mask(path, size=(600, 600)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")

    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


# -------------------------------------------------
# Visualization
# -------------------------------------------------
def save_comparison(img, gt_mask, pred_mask, title, save_path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original CXR")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------
# Run on Dataset
# -------------------------------------------------
cxr_dir = "../data/COVIDQU/Lung Segmentation Data/Test/COVID-19/images"
mask_dir = "../data/COVIDQU/Lung Segmentation Data/Test/COVID-19/lung masks"
# cxr_dir = "JSRT/jsrt/cxr"
# mask_dir = "JSRT/jsrt/masks"

output_dir = "outputs3"
os.makedirs(output_dir, exist_ok=True)

image_files = sorted([
    f for f in os.listdir(cxr_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])[:5]

for fname in image_files:
    cxr_path = os.path.join(cxr_dir, fname)
    mask_path = os.path.join(mask_dir, fname)

    if not os.path.exists(mask_path):
        print(f"[WARNING] Mask not found for {fname}")
        continue

    img = read_and_preprocess(cxr_path)
    gt_mask = read_mask(mask_path)

    pred_mask, _ = lung_segmentation_pipeline_from_image(img)

    save_path = os.path.join(
        output_dir, f"JSRT_LungSeg_{os.path.splitext(fname)[0]}.png"
    )

    save_comparison(
        img,
        gt_mask,
        pred_mask,
        title=f"JSRT Lung Segmentation: {fname}",
        save_path=save_path
    )

    print(f"[SAVED] {save_path}")
