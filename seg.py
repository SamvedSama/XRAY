import os
import random
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.filters import threshold_otsu, sobel, scharr
from skimage.feature import canny
from skimage.segmentation import flood, watershed
from skimage import img_as_float
from scipy import ndimage as ndi
from tqdm import tqdm
import pandas as pd


# -------------------- Utility --------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def dice_score(pred, gt):
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)

def iou_score(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / (union + 1e-8)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float() / 255.0
    return img, img_t

def postprocess_mask(mask_tensor):
    mask = mask_tensor.squeeze().detach().cpu().numpy()
    return (mask > 0.5).astype(np.uint8)

def save_comparison(original, pred, gt_path, save_path):
    """
    Creates a side-by-side comparison of original, ground truth (if available), and predicted mask.
    Works safely with grayscale or RGB images.
    """
    # Ensure prediction is in uint8 [0,255]
    pred = (pred * 255).astype(np.uint8) if pred.max() <= 1 else pred.astype(np.uint8)

    # Convert all to same size
    orig_resized = cv2.resize(original, (pred.shape[1], pred.shape[0]))

    # Ensure 3 channels for all
    if len(orig_resized.shape) == 2:
        orig_resized = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)

    if len(pred.shape) == 2:
        pred_color = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
    else:
        pred_color = pred

    if gt_path and os.path.exists(gt_path):
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
        gt_color = cv2.applyColorMap((gt_resized).astype(np.uint8), cv2.COLORMAP_JET)
        combined = np.hstack([orig_resized, gt_color, pred_color])
    else:
        combined = np.hstack([orig_resized, pred_color])

    cv2.imwrite(save_path, combined)




# -------------------- Segmentation Models --------------------

def get_models():
    models = {
        "UNet": smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1),
        "ResUNet": smp.Unet("resnet50", encoder_weights="imagenet", in_channels=1, classes=1),
        "ResNet34": smp.FPN("resnet34", encoder_weights="imagenet", in_channels=1, classes=1),
        "DeepLabV3+": smp.DeepLabV3Plus("resnet34", encoder_weights="imagenet", in_channels=1, classes=1),
        "FPN": smp.FPN("resnet50", encoder_weights="imagenet", in_channels=1, classes=1),
        "PSPNet": smp.PSPNet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
    }
    for m in models.values():
        m.eval()
    return models


# -------------------- Preprocessing Methods --------------------

def edge_detect(img, method):
    img_f = img_as_float(img)
    if method == "canny":
        return canny(img_f, sigma=1)
    elif method == "sobel":
        return sobel(img_f)
    elif method == "scharr":
        return scharr(img_f)
    return img_f

def clusterize(img, method, n=3):
    h, w = img.shape
    flat = img.reshape(-1, 1)
    if method == "kmeans":
        clt = MiniBatchKMeans(n_clusters=n, batch_size=1024, n_init=5)
    else:
        clt = KMeans(n_clusters=n, n_init=5)
    labels = clt.fit_predict(flat)
    return labels.reshape(h, w)

def region_based(img, method):
    gray = cv2.GaussianBlur(img, (5, 5), 0)
    if method == "watershed":
        gradient = sobel(gray)
        markers, _ = ndi.label(gradient < 0.05)
        segmented = watershed(gradient, markers)
        return (segmented > 1).astype(np.uint8)
    elif method == "grabcut":
        mask = np.zeros(gray.shape, np.uint8)
        rect = (10, 10, gray.shape[1] - 20, gray.shape[0] - 20)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    elif method == "region_grow":
        seed = (gray.shape[0]//2, gray.shape[1]//2)
        mask = flood(gray, seed_point=seed, tolerance=0.05)
        return mask.astype(np.uint8)
    return gray

def thresholding(img, method="otsu"):
    if method == "otsu":
        t = threshold_otsu(img)
        return (img > t).astype(np.uint8)
    elif method == "adaptive":
        return cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return img


# -------------------- Segmentation Function --------------------

def segment(model, x):
    with torch.no_grad():
        pred = model(x)
    return postprocess_mask(pred)


# -------------------- Stage Runner --------------------

def run_stage(stage_name, preprocessors, dataset_name, img_dir, mask_dir, out_dir, num_samples=5):
    ensure_dir(out_dir)
    models = get_models()
    results = []

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    img_files = random.sample(img_files, min(num_samples, len(img_files)))

    for img_name in tqdm(img_files, desc=f"{dataset_name} - {stage_name}"):
        img_path = os.path.join(img_dir, img_name)
        original, img_t = preprocess_image(img_path)

        for prep_name, prep_func in preprocessors.items():
            pre_img = prep_func(original) if prep_func else original
            if pre_img.ndim == 2:
                x = torch.tensor(pre_img).unsqueeze(0).unsqueeze(0).float()
            else:
                x = torch.tensor(pre_img).permute(2, 0, 1).unsqueeze(0).float()

            for model_name, model in models.items():
                pred = segment(model, x)
                gt_path = os.path.join(mask_dir, img_name) if mask_dir and os.path.exists(mask_dir) else None
                save_name = f"{os.path.splitext(img_name)[0]}_{prep_name}_{model_name}.png"
                save_path = os.path.join(out_dir, save_name)
                save_comparison(original, pred, gt_path, save_path)

                if gt_path and os.path.exists(gt_path):
                    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    gt = cv2.resize(gt, (256, 256))
                    gt = (gt > 127).astype(np.uint8)
                    iou = iou_score(pred, gt)
                    dice = dice_score(pred, gt)
                    results.append((dataset_name, img_name, prep_name, model_name, iou, dice))
    return results


# -------------------- Main Runner --------------------

def run_all_stages(dataset_name, img_dir, mask_dir, out_dir):
    results = []

    # 1️⃣ Raw
    results += run_stage("Raw", {"none": None}, dataset_name, img_dir, mask_dir, os.path.join(out_dir, "Raw"))

    # 2️⃣ Edge-based
    results += run_stage("Edge", {"canny": lambda x: edge_detect(x, "canny"),
                                  "sobel": lambda x: edge_detect(x, "sobel")},
                         dataset_name, img_dir, mask_dir, os.path.join(out_dir, "Edge"))

    # 3️⃣ Cluster-based
    results += run_stage("Cluster", {"kmeans": lambda x: clusterize(x, "kmeans"),
                                     "mbkmeans": lambda x: clusterize(x, "minibatch")},
                         dataset_name, img_dir, mask_dir, os.path.join(out_dir, "Cluster"))

    # 4️⃣ Region-based
    results += run_stage("Region", {"watershed": lambda x: region_based(x, "watershed"),
                                    "grabcut": lambda x: region_based(x, "grabcut"),
                                    "region_grow": lambda x: region_based(x, "region_grow")},
                         dataset_name, img_dir, mask_dir, os.path.join(out_dir, "Region"))

    # 5️⃣ Threshold-based
    results += run_stage("Threshold", {"otsu": lambda x: thresholding(x, "otsu"),
                                       "adaptive": lambda x: thresholding(x, "adaptive")},
                         dataset_name, img_dir, mask_dir, os.path.join(out_dir, "Threshold"))

    return results


# -------------------- Dataset Walker --------------------

def main():
    base_dir = "data"
    output_root = "results_full_comparison"
    ensure_dir(output_root)
    all_results = []

    datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for dataset_name in datasets:
        dataset_path = os.path.join(base_dir, dataset_name)

        if dataset_name == "synth":
            synth_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
            for s in synth_folders:
                img_dir = os.path.join(dataset_path, s)
                mask_dir = None
                out_dir = os.path.join(output_root, "synth", s)
                all_results += run_all_stages(f"synth_{s}", img_dir, mask_dir, out_dir)
        elif dataset_name == "COVIDQU":
            for sub in ["Infection Segmentation Data", "Lung Segmentation Data"]:
                sub_path = os.path.join(dataset_path, sub)
                for split in ["Train", "Test", "Val"]:
                    split_path = os.path.join(sub_path, split)
                    for cls in ["COVID-19", "Non-COVID", "Normal"]:
                        cls_path = os.path.join(split_path, cls)
                        if not os.path.exists(cls_path):
                            continue
                        img_dir, mask_dir = None, None
                        for root, dirs, files in os.walk(cls_path):
                            if "image" in root.lower():
                                img_dir = root
                            if "mask" in root.lower():
                                mask_dir = root
                        if img_dir:
                            out_dir = os.path.join(output_root, f"COVIDQU_{sub}_{split}_{cls}")
                            all_results += run_all_stages(f"COVIDQU_{sub}_{split}_{cls}", img_dir, mask_dir, out_dir)
        else:
            img_dir, mask_dir = None, None
            for root, dirs, files in os.walk(dataset_path):
                if "image" in root.lower():
                    img_dir = root
                if "mask" in root.lower():
                    mask_dir = root
            if img_dir:
                out_dir = os.path.join(output_root, dataset_name)
                all_results += run_all_stages(dataset_name, img_dir, mask_dir, out_dir)

    df = pd.DataFrame(all_results, columns=["Dataset", "Image", "Preprocessing", "Model", "IoU", "Dice"])
    df.to_csv(os.path.join(output_root, "metrics_summary.csv"), index=False)
    print("\n✅ All processing done. Results saved in:", output_root)


if __name__ == "__main__":
    main()
