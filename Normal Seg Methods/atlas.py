import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology
from scipy.spatial.distance import cdist
import SimpleITK as sitk


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    return (2.0 * intersection) / (np.sum(gt) + np.sum(pred) + 1e-8)


def jaccard_index(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    return intersection / (union + 1e-8)


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

    return np.percentile(np.hstack([d1, d2]), 95)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def preprocess_for_registration(img, size=(600, 600)):
    img = cv2.resize(img, size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def read_mask(path, size=(600, 600)):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def read_image(path, size=(600, 600)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, size)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


# ---------------------------------------------------------
# Saving Visual Outputs
# ---------------------------------------------------------
def save_with_gt(img, pred, gt, title, save_path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Atlas Prediction")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_without_gt(img, pred, title, save_path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap="gray")
    plt.title("Atlas Mask")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---------------------------------------------------------
# ATLAS SEGMENTATION PIPELINE
# ---------------------------------------------------------
def atlas_based_lung_segmentation(target_img, atlas_img, atlas_mask):

    # Preprocess
    target = preprocess_for_registration(target_img)
    atlas = preprocess_for_registration(atlas_img)

    atlas_mask = atlas_mask.astype(np.uint8)
    mask = cv2.resize(atlas_mask, target.shape[::-1], interpolation=cv2.INTER_NEAREST)

    H, W = target.shape
    y1, y2 = int(0.20 * H), int(0.80 * H)
    x1, x2 = int(0.20 * W), int(0.80 * W)

    t_crop = target[y1:y2, x1:x2]
    a_crop = atlas[y1:y2, x1:x2]
    m_crop = mask[y1:y2, x1:x2]

    t_sitk = sitk.GetImageFromArray(t_crop.astype(np.float32))
    a_sitk = sitk.GetImageFromArray(a_crop.astype(np.float32))
    m_sitk = sitk.GetImageFromArray(m_crop.astype(np.uint8))

    composite_tx = sitk.CompositeTransform(2)

    # ------------------ RIGID ---------------------
    rigid_reg = sitk.ImageRegistrationMethod()
    rigid_reg.SetMetricAsMattesMutualInformation(40)
    rigid_reg.SetMetricSamplingPercentage(0.2)
    rigid_reg.SetMetricSamplingStrategy(rigid_reg.RANDOM)
    rigid_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=0.05, numberOfIterations=15
    )
    rigid_reg.SetInterpolator(sitk.sitkLinear)
    rigid_reg.SetInitialTransform(sitk.CenteredTransformInitializer(
        t_sitk, a_sitk, sitk.Euler2DTransform()
    ), inPlace=False)

    try:
        composite_tx.AddTransform(rigid_reg.Execute(t_sitk, a_sitk))
    except:
        pass

    # ------------------ AFFINE --------------------
    aff_reg = sitk.ImageRegistrationMethod()
    aff_reg.SetMetricAsMattesMutualInformation(30)
    aff_reg.SetMetricSamplingPercentage(0.2)
    aff_reg.SetMetricSamplingStrategy(aff_reg.RANDOM)
    aff_reg.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=20)
    aff_reg.SetInterpolator(sitk.sitkLinear)
    aff_reg.SetInitialTransform(sitk.AffineTransform(2), inPlace=False)

    try:
        composite_tx.AddTransform(aff_reg.Execute(t_sitk, a_sitk))
    except:
        pass

    # ------------------ B-SPLINE ------------------
    bs_reg = sitk.ImageRegistrationMethod()
    bs_reg.SetMetricAsMattesMutualInformation(20)
    bs_reg.SetMetricSamplingPercentage(0.15)
    bs_reg.SetMetricSamplingStrategy(bs_reg.RANDOM)
    bs_reg.SetOptimizerAsGradientDescent(learningRate=0.3, numberOfIterations=10)
    bs_reg.SetInterpolator(sitk.sitkLinear)

    try:
        bs_init = sitk.BSplineTransformInitializer(t_sitk, [10, 10])
        bs_reg.SetInitialTransform(bs_init, inPlace=False)
        composite_tx.AddTransform(bs_reg.Execute(t_sitk, a_sitk))
    except:
        pass

    # ------------------ WARP ----------------------
    warped_crop = sitk.Resample(
        m_sitk, t_sitk, composite_tx,
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )
    warped_crop = sitk.GetArrayFromImage(warped_crop)

    full_mask = np.zeros_like(target)
    full_mask[y1:y2, x1:x2] = warped_crop

    # Post-processing
    m = (full_mask > 0).astype(np.uint8)
    m = morphology.binary_opening(m, morphology.disk(5))
    m = binary_fill_holes(m).astype(np.uint8)

    labels = measure.label(m)
    props = sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)

    clean = np.zeros_like(m)
    for r in props[:2]:
        clean[labels == r.label] = 1

    clean = morphology.binary_closing(clean, morphology.disk(8))
    clean = morphology.binary_opening(clean, morphology.disk(5))

    return clean.astype(np.uint8)


# ---------------------------------------------------------
# PROCESS DATASET
# ---------------------------------------------------------
def process_dataset(dataset_name, image_path, mask_path, atlas_img, atlas_mask):

    print(f"\n=== Processing {dataset_name} Dataset ===")

    out_dir = os.path.join("outputs_ATLAS", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    metrics_file = open(os.path.join(out_dir, "metrics.txt"), "w")
    metrics_file.write(f"=== METRICS FOR {dataset_name} ===\n\n")

    dice_vals, jacc_vals, hd95_vals = [], [], []

    files = sorted([f for f in os.listdir(image_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    for fname in files:
        img = read_image(os.path.join(image_path, fname))
        pred = atlas_based_lung_segmentation(img, atlas_img, atlas_mask)

        save_path = os.path.join(out_dir, f"{dataset_name}_{fname}.png")

        if mask_path is None:
            save_without_gt(img, pred, fname, save_path)
            metrics_file.write(f"{fname}: NO GT\n")
            continue

        gt = read_mask(os.path.join(mask_path, fname))

        save_with_gt(img, pred, gt, fname, save_path)

        dsc = dice_score(gt, pred)
        jacc = jaccard_index(gt, pred)
        hd = hausdorff95(gt, pred)

        dice_vals.append(dsc)
        jacc_vals.append(jacc)
        hd95_vals.append(hd)

        metrics_file.write(f"{fname}: DSC={dsc:.4f}, IoU={jacc:.4f}, HD95={hd:.2f}\n")
        print(f"{fname}: DSC={dsc:.4f}, IoU={jacc:.4f}, HD95={hd:.2f}")
        print(f"[SAVED] {save_path}")

    if mask_path is not None:
        metrics_file.write("\n--- AVERAGES ---\n")
        metrics_file.write(f"Mean DSC: {np.nanmean(dice_vals):.4f}\n")
        metrics_file.write(f"Mean IoU: {np.nanmean(jacc_vals):.4f}\n")
        metrics_file.write(f"Mean HD95: {np.nanmean(hd95_vals):.2f}\n")

    metrics_file.close()
    print(f"[METRICS SAVED] outputs_ATLAS/{dataset_name}/metrics.txt")


# ---------------------------------------------------------
# LOAD ATLAS
# ---------------------------------------------------------
ATLAS_IMAGE_PATH = "Used/JSRT/lung/JPCLN001.png"
ATLAS_MASK_PATH = "Used/JSRT/mask/JPCLN001.png"

atlas_img = read_image(ATLAS_IMAGE_PATH)
atlas_mask = read_mask(ATLAS_MASK_PATH)


# ---------------------------------------------------------
# RUN ALL DATASETS
# ---------------------------------------------------------
process_dataset("COVIDQU", "Used/COVIDQU/lung", "Used/COVIDQU/mask", atlas_img, atlas_mask)
process_dataset("JSRT",     "Used/JSRT/lung",     "Used/JSRT/mask", atlas_img, atlas_mask)
process_dataset("PA",       "Used/pa/lung",       "Used/pa/mask",  atlas_img, atlas_mask)
process_dataset("SYNTH",    "Used/synth",          None,            atlas_img, atlas_mask)
