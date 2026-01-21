import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes

import SimpleITK as sitk


# ---------------------------------------------------------
# Helper Conversions
# ---------------------------------------------------------
def cv2_to_sitk(img):
    return sitk.GetImageFromArray(img.astype(np.float32))


def sitk_to_cv2(img):
    return sitk.GetArrayFromImage(img).astype(np.uint8)


# ---------------------------------------------------------
# Preprocessing for both atlas + target
# ---------------------------------------------------------
def preprocess_for_registration(img, size=(600, 600)):
    img = cv2.resize(img, size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (5,5), 0)
    return img


# ---------------------------------------------------------
# ATLAS-BASED SEGMENTATION
# ---------------------------------------------------------
# def atlas_based_lung_segmentation(target_img, atlas_img, atlas_mask):
#     """
#     Correct classical atlas-based lung segmentation:
#     Rigid registration + BSpline deformable registration
#     with proper transform composition.
#     """

#     # ----------------------------------------
#     # 1. Preprocess
#     # ----------------------------------------
#     target = preprocess_for_registration(target_img)
#     atlas  = preprocess_for_registration(atlas_img)
#     mask   = cv2.resize(atlas_mask, target.shape[::-1], interpolation=cv2.INTER_NEAREST)

#     # ----------------------------------------
#     # 2. Crop central chest region
#     # ----------------------------------------
#     H, W = target.shape
#     y1, y2 = int(0.15 * H), int(0.85 * H)
#     x1, x2 = int(0.15 * W), int(0.85 * W)

#     t_crop = target[y1:y2, x1:x2]
#     a_crop = atlas[y1:y2, x1:x2]
#     m_crop = mask[y1:y2, x1:x2]

#     t_sitk = cv2_to_sitk(t_crop)
#     a_sitk = cv2_to_sitk(a_crop)
#     m_sitk = cv2_to_sitk(m_crop)

#     # ----------------------------------------
#     # 3. RIGID REGISTRATION
#     # ----------------------------------------
#     rigid_reg = sitk.ImageRegistrationMethod()
#     rigid_reg.SetMetricAsMattesMutualInformation(50)
#     rigid_reg.SetInterpolator(sitk.sitkLinear)

#     rigid_reg.SetOptimizerAsRegularStepGradientDescent(
#         learningRate=1.0,
#         minStep=0.001,
#         numberOfIterations=50
#     )

#     rigid_init = sitk.CenteredTransformInitializer(
#         t_sitk, a_sitk, sitk.Euler2DTransform()
#     )
#     rigid_reg.SetInitialTransform(rigid_init, inPlace=False)

#     rigid_tx = rigid_reg.Execute(t_sitk, a_sitk)

#     # ----------------------------------------
#     # 4. BSPLINE DEFORMABLE REGISTRATION
#     # ----------------------------------------
#     bspline_reg = sitk.ImageRegistrationMethod()
#     bspline_reg.SetMetricAsMattesMutualInformation(50)
#     bspline_reg.SetInterpolator(sitk.sitkLinear)

#     bspline_reg.SetOptimizerAsGradientDescent(
#         learningRate=0.2,
#         numberOfIterations=40
#     )

#     bspline_tx = sitk.BSplineTransformInitializer(
#         t_sitk, transformDomainMeshSize=[30, 30]
#     )

#     bspline_reg.SetInitialTransform(bspline_tx, inPlace=False)

#     bspline_tx_final = bspline_reg.Execute(t_sitk, a_sitk)

#     # ----------------------------------------
#     # 5. COMPOSE TRANSFORMS (CORRECT WAY)
#     # ----------------------------------------
#     composite_tx = sitk.CompositeTransform(2)
#     composite_tx.AddTransform(rigid_tx)
#     composite_tx.AddTransform(bspline_tx_final)

#     # ----------------------------------------
#     # 6. Warp atlas mask
#     # ----------------------------------------
#     warped = sitk.Resample(
#         m_sitk,
#         t_sitk,
#         composite_tx,
#         sitk.sitkNearestNeighbor,
#         0,
#         sitk.sitkUInt8
#     )

#     warped_crop = sitk_to_cv2(warped)

#     # Uncrop to full image
#     full_mask = np.zeros_like(target)
#     full_mask[y1:y2, x1:x2] = warped_crop

#     # ----------------------------------------
#     # 7. Post-processing
#     # ----------------------------------------
#     full_mask = (full_mask > 0).astype(np.uint8)
#     full_mask = binary_fill_holes(full_mask).astype(np.uint8)

#     labels = measure.label(full_mask, connectivity=2)
#     regions = sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)

#     final_mask = np.zeros_like(full_mask)
#     for r in regions[:2]:
#         final_mask[labels == r.label] = 1

#     final_mask = morphology.binary_closing(final_mask, morphology.disk(10))
#     final_mask = morphology.binary_opening(final_mask, morphology.disk(5))

#     return final_mask.astype(np.uint8)



def atlas_based_lung_segmentation(target_img, atlas_img, atlas_mask):
    """
    Stable & fast atlas segmentation:
       - CLAHE preprocessing
       - Chest crop
       - Rigid + Affine light registration
       - Optional BSpline refinement
       - Hard iteration limits + sampling
       - Fallback to rigid warp on failure
    """

    # ----------------------------------------
    # Preprocess both images
    # ----------------------------------------
    target = preprocess_for_registration(target_img)
    atlas  = preprocess_for_registration(atlas_img)
    mask   = cv2.resize(atlas_mask, target.shape[::-1], interpolation=cv2.INTER_NEAREST)

    H, W = target.shape

    # ----------------------------------------
    # Crop middle chest region
    # ----------------------------------------
    y1, y2 = int(0.20 * H), int(0.80 * H)
    x1, x2 = int(0.20 * W), int(0.80 * W)

    t_crop = target[y1:y2, x1:x2]
    a_crop = atlas[y1:y2, x1:x2]
    m_crop = mask[y1:y2, x1:x2]

    t_sitk = sitk.GetImageFromArray(t_crop.astype(np.float32))
    a_sitk = sitk.GetImageFromArray(a_crop.astype(np.float32))
    m_sitk = sitk.GetImageFromArray(m_crop.astype(np.uint8))

    # Final combined transform
    composite_tx = sitk.CompositeTransform(2)

    # -------------------------------------------------
    # 🔹 1. RIGID REGISTRATION  (FAST + SAFE)
    # -------------------------------------------------
    rigid_reg = sitk.ImageRegistrationMethod()
    rigid_reg.SetMetricAsMattesMutualInformation(40)
    rigid_reg.SetMetricSamplingStrategy(rigid_reg.RANDOM)
    rigid_reg.SetMetricSamplingPercentage(0.20)
    rigid_reg.SetInterpolator(sitk.sitkLinear)

    rigid_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.05,
        numberOfIterations=15,   # small, safe
        relaxationFactor=0.5
    )

    rigid_reg.SetShrinkFactorsPerLevel([2,1])
    rigid_reg.SetSmoothingSigmasPerLevel([2,1])

    rigid_init = sitk.CenteredTransformInitializer(
        t_sitk,
        a_sitk,
        sitk.Euler2DTransform()
    )
    rigid_reg.SetInitialTransform(rigid_init, inPlace=False)

    try:
        rigid_tx = rigid_reg.Execute(t_sitk, a_sitk)
        composite_tx.AddTransform(rigid_tx)
    except:
        print("[WARN] Rigid registration failed → using identity")
        pass

    # -------------------------------------------------
    # 🔹 2. AFFINE REGISTRATION (FAST)
    # -------------------------------------------------
    aff_reg = sitk.ImageRegistrationMethod()
    aff_reg.SetMetricAsMattesMutualInformation(30)
    aff_reg.SetMetricSamplingStrategy(aff_reg.RANDOM)
    aff_reg.SetMetricSamplingPercentage(0.20)
    aff_reg.SetInterpolator(sitk.sitkLinear)

    aff_reg.SetOptimizerAsGradientDescent(
        learningRate=0.5,
        numberOfIterations=20
    )

    aff_reg.SetShrinkFactorsPerLevel([2,1])
    aff_reg.SetSmoothingSigmasPerLevel([2,1])

    aff_init = sitk.AffineTransform(2)
    aff_reg.SetInitialTransform(aff_init, inPlace=False)

    try:
        aff_tx = aff_reg.Execute(t_sitk, a_sitk)
        composite_tx.AddTransform(aff_tx)
    except:
        print("[WARN] Affine registration failed → skip")

    # -------------------------------------------------
    # 🔹 3. OPTIONAL B-SPLINE (VERY LIGHT)
    # -------------------------------------------------
    bs_reg = sitk.ImageRegistrationMethod()
    bs_reg.SetMetricAsMattesMutualInformation(20)
    bs_reg.SetMetricSamplingStrategy(bs_reg.RANDOM)
    bs_reg.SetMetricSamplingPercentage(0.15)
    bs_reg.SetInterpolator(sitk.sitkLinear)

    bs_reg.SetOptimizerAsGradientDescent(
        learningRate=0.3,
        numberOfIterations=10   # very small
    )

    try:
        bs_init = sitk.BSplineTransformInitializer(t_sitk, [10, 10])
        bs_reg.SetInitialTransform(bs_init, inPlace=False)
        bs_tx = bs_reg.Execute(t_sitk, a_sitk)
        composite_tx.AddTransform(bs_tx)
    except:
        print("[WARN] BSpline failed → skipping")

    # -------------------------------------------------
    # 🔹 4. WARP THE ATLAS MASK
    # -------------------------------------------------
    warped_crop = sitk.Resample(
        m_sitk,
        t_sitk,
        composite_tx,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )
    warped_crop = sitk.GetArrayFromImage(warped_crop)

    # Uncrop back
    full_mask = np.zeros_like(target)
    full_mask[y1:y2, x1:x2] = warped_crop

    # -------------------------------------------------
    # 🔹 5. STRONG POST-PROCESSING
    # -------------------------------------------------
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
# Mask loader (for ground truth)
# ---------------------------------------------------------
def read_mask(path, size=(600, 600)):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


# ---------------------------------------------------------
# Image loader
# ---------------------------------------------------------
def read_image(path, size=(600, 600)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, size)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


# ---------------------------------------------------------
# Saving comparison (for datasets with masks)
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


# ---------------------------------------------------------
# Saving result only (for synthetic dataset)
# ---------------------------------------------------------
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
# Dataset Processing
# ---------------------------------------------------------
def process_dataset(dataset_name, image_path, mask_path, atlas_img, atlas_mask):
    print(f"\n=== Processing {dataset_name} Dataset ===")

    out_dir = os.path.join("outputs_ATLAS", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(image_path) if f.lower().endswith((".png",".jpg",".jpeg"))])

    for fname in files:
        img = read_image(os.path.join(image_path, fname))

        pred_mask = atlas_based_lung_segmentation(img, atlas_img, atlas_mask)

        save_path = os.path.join(out_dir, f"{dataset_name}_LungSeg_{fname}_ATLAS.png")

        if mask_path is not None:
            gt = read_mask(os.path.join(mask_path, fname))
            if gt is not None:
                save_with_gt(img, pred_mask, gt, f"{dataset_name}: {fname}", save_path)
            else:
                save_without_gt(img, pred_mask, f"{dataset_name}: {fname}", save_path)
        else:
            save_without_gt(img, pred_mask, f"{dataset_name}: {fname}", save_path)

        print(f"[SAVED] {save_path}")


# ---------------------------------------------------------
# CONFIGURE YOUR ATLAS HERE
# ---------------------------------------------------------
# Use a CLEAN JSRT image + lung mask as the atlas
ATLAS_IMAGE_PATH = "Used/JSRT/lung/JPCLN001.png"   # sample
ATLAS_MASK_PATH  = "Used/JSRT/mask/JPCLN001.png"

atlas_img  = read_image(ATLAS_IMAGE_PATH)
atlas_mask = read_mask(ATLAS_MASK_PATH)


# ---------------------------------------------------------
# RUN ALL DATASETS
# ---------------------------------------------------------
process_dataset(
    "COVIDQU",
    image_path="Used/COVIDQU/lung",
    mask_path="Used/COVIDQU/mask",
    atlas_img=atlas_img,
    atlas_mask=atlas_mask
)

process_dataset(
    "JSRT",
    image_path="Used/JSRT/lung",
    mask_path="Used/JSRT/mask",
    atlas_img=atlas_img,
    atlas_mask=atlas_mask
)

process_dataset(
    "synth",
    image_path="Used/synth",
    mask_path=None,
    atlas_img=atlas_img,
    atlas_mask=atlas_mask
)
