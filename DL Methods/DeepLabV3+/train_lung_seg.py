"""
Lung Segmentation from Chest X-Ray Images
Replicating: "Lung Segmentation from Chest X-Ray Images Using DeepLabV3Plus-Based CNN Model"
Indonesian Journal of Computer Science, Vol. 13, No. 1, 2024

Model   : DeepLabV3+ with ResNet-50 encoder (ImageNet pretrained)
Loss    : DiceBCE (combined)
Metrics : DSC, IoU, Sensitivity, Specificity, MCC, Cohen's Kappa, HD95, NSD
Hardware: Tuned for RTX 4060 Ti 8GB, 32GB RAM

Directory layout expected:
    Standard_Data/
    ├── metadata_with_generations_cleaned.csv
    ├── train/
    │   ├── covid-19/   (cxr/, masks/)
    │   ├── indiana/    (cxr/, masks/)
    │   ├── jsrt/       (cxr/, masks/)
    │   ├── normal/     (cxr/, masks/)
    │   ├── pneumonia/  (cxr/, masks/)
    │   └── tb/         (cxr/, masks/)
    ├── val/
    │   ├── covid19/    (cxr/, masks/)
    │   ├── jsrt/       (cxr/, masks/)
    │   ├── normal/     (cxr/, masks/)
    │   ├── pneumonia/  (cxr/, masks/)
    │   └── tb/         (cxr/, masks/)
    └── test/
        ├── covid19/    (cxr/, masks/)
        ├── normal/     (cxr/, masks/)
        ├── synth/      (cxr/ ONLY — no masks)
        └── tb/         (cxr/, masks/)
"""

import os
import gc
import copy
import random
import re

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

from scipy.ndimage import distance_transform_edt
from skimage import morphology


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
SEED        = 42
ROOT        = "f:/Capstone/Standard_Data"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS      = 15
LR          = 1e-3
BATCH_SIZE  = 4        # RTX 4060 Ti 8 GB handles batch=4 at 256×256 comfortably
IMAGE_SIZE  = 256
USE_CLAHE   = True

# AMP (Automatic Mixed Precision) — halves VRAM, speeds up training on Ampere+
USE_AMP     = True

# Early stopping — stop if val Dice does not improve for this many epochs
EARLY_STOP_PATIENCE = 5

# Supported image extensions (covers .png, .jpg, .jpeg, .bmp used in your dataset)
IMG_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Output directories
OUTPUT_DIR      = "outputs"
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
WEIGHTS_DIR     = os.path.join(OUTPUT_DIR, "weights")
LOG_DIR         = os.path.join(OUTPUT_DIR, "logs")
PRED_DIR        = os.path.join(OUTPUT_DIR, "predictions")
TRAIN_PRED_DIR  = os.path.join(PRED_DIR, "train")
VAL_PRED_DIR    = os.path.join(PRED_DIR, "val")
TEST_PRED_DIR   = os.path.join(PRED_DIR, "test")
SYNTH_PRED_DIR  = os.path.join(PRED_DIR, "test_synth")

for d in [
    OUTPUT_DIR, CHECKPOINT_DIR, WEIGHTS_DIR, LOG_DIR,
    PRED_DIR, TRAIN_PRED_DIR, VAL_PRED_DIR,
    TEST_PRED_DIR, SYNTH_PRED_DIR,
]:
    os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────────────────────
# GPU CLEANUP
# ──────────────────────────────────────────────────────────────
def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────
# DATA COLLECTION
# ──────────────────────────────────────────────────────────────
def _is_image(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in IMG_EXTS


def collect_split(split: str):
    """
    Walk   ROOT/<split>/<category>/cxr/   and match to
           ROOT/<split>/<category>/masks/ by filename (stem-insensitive).

    Returns:
        imgs  : list[str]  — absolute paths to CXR images
        masks : list[str]  — corresponding mask paths (same order)

    Only pairs where BOTH the image AND its mask exist are included.
    Categories that lack a masks/ folder are silently skipped
    (use collect_split_no_gt for those).
    """
    split_dir = os.path.join(ROOT, split)
    imgs, masks = [], []

    if not os.path.isdir(split_dir):
        print(f"  [WARN] split directory not found: {split_dir}")
        return imgs, masks

    for category in sorted(os.listdir(split_dir)):
        cxr_dir  = os.path.join(split_dir, category, "cxr")
        mask_dir = os.path.join(split_dir, category, "masks")

        # Skip categories without masks (e.g. test/synth)
        if not os.path.isdir(cxr_dir) or not os.path.isdir(mask_dir):
            continue

        # Build a stem→path lookup for masks so we can match by filename
        # regardless of extension (e.g. .bmp image paired with .png mask)
        mask_lookup: dict[str, str] = {}
        for mf in os.listdir(mask_dir):
            if _is_image(mf):
                stem = os.path.splitext(mf)[0]
                mask_lookup[stem] = os.path.join(mask_dir, mf)

        # Build a numeric-ID → mask path lookup for pneumonia-style naming
        id_lookup: dict[str, str] = {}
        for mpath in mask_lookup.values():
            mstem = os.path.splitext(os.path.basename(mpath))[0]
            m = re.search(r'(\d+)$', mstem)
            if m:
                id_lookup[m.group(1)] = mpath

        for cf in sorted(os.listdir(cxr_dir)):
            if not _is_image(cf):
                continue
            stem     = os.path.splitext(cf)[0]
            img_path = os.path.join(cxr_dir, cf)

            if stem in mask_lookup:
                imgs.append(img_path)
                masks.append(mask_lookup[stem])
            else:
                # Exact-filename fallback (mask has identical extension)
                mask_path = os.path.join(mask_dir, cf)
                if os.path.isfile(mask_path):
                    imgs.append(img_path)
                    masks.append(mask_path)
                else:
                    # Trailing numeric ID fallback (e.g. CXR_Images_001 → CXR_Mask_001)
                    m = re.search(r'(\d+)$', stem)
                    if m and m.group(1) in id_lookup:
                        imgs.append(img_path)
                        masks.append(id_lookup[m.group(1)])
                    # If still no match, the image is silently dropped (no GT)

    return imgs, masks


def collect_split_no_gt(split: str, category: str):
    """
    Collect images from ROOT/<split>/<category>/cxr/ that have NO masks.
    Used for test/synth.
    """
    cxr_dir = os.path.join(ROOT, split, category, "cxr")
    if not os.path.isdir(cxr_dir):
        print(f"  [WARN] directory not found: {cxr_dir}")
        return []
    return sorted(
        os.path.join(cxr_dir, f)
        for f in os.listdir(cxr_dir)
        if _is_image(f)
    )


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class LungDataset(Dataset):
    """
    With masks    → (img_tensor, mask_tensor, orig_np, name, img_path, mask_path)
    Without masks → (img_tensor, orig_np, name, img_path)
    """
    def __init__(self, images, masks=None, image_size=256, use_clahe=False):
        self.images     = images
        self.masks      = masks
        self.image_size = image_size
        self.use_clahe  = use_clahe

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_name = os.path.basename(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            from PIL import Image
            img = np.array(Image.open(img_path).convert("L"))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.resize(img, (self.image_size, self.image_size))

        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img   = clahe.apply(img)

        img_tensor = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)

        if self.masks is not None:
            mask_path = self.masks[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                from PIL import Image
                mask = np.array(Image.open(mask_path).convert("L"))
            if mask is None:
                raise FileNotFoundError(f"Cannot read mask: {mask_path}")
            mask        = cv2.resize(mask, (self.image_size, self.image_size))
            mask        = (mask > 127).astype("float32")
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return img_tensor, mask_tensor, img, img_name, img_path, mask_path
        else:
            return img_tensor, img, img_name, img_path


# ──────────────────────────────────────────────────────────────
# LOSS
# ──────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    """BCE + Dice — same as paper's training objective."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sig  = torch.sigmoid(pred)
        smooth    = 1e-8
        inter     = (pred_sig * target).sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2.0 * inter + smooth) /
                         (pred_sig.sum(dim=(1, 2, 3)) +
                          target.sum(dim=(1, 2, 3)) + smooth)).mean()

        return bce_loss + dice_loss


# ──────────────────────────────────────────────────────────────
# QUICK BATCH METRICS  (used during training loop)
# ──────────────────────────────────────────────────────────────
def batch_metrics(pred_logits, mask):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    TP = (pred * mask).sum()
    TN = ((1 - pred) * (1 - mask)).sum()
    FP = (pred * (1 - mask)).sum()
    FN = ((1 - pred) * mask).sum()
    acc  = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    dice = (2 * TP)  / (2 * TP + FP + FN + 1e-8)
    iou  = TP        / (TP + FP + FN + 1e-8)
    return acc.item(), dice.item(), iou.item()


# ──────────────────────────────────────────────────────────────
# 8-METRIC SUITE  (used at evaluation / save_predictions time)
# ──────────────────────────────────────────────────────────────
def dice_score(gt, pred):
    inter = np.sum(gt * pred)
    return (2.0 * inter) / (np.sum(gt) + np.sum(pred) + 1e-8)

def jaccard_index(gt, pred):
    inter = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - inter
    return inter / (union + 1e-8)

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
    n  = tp + tn + fp + fn
    po = (tp + tn) / (n + 1e-8)
    pe = (((tp + fn) / (n + 1e-8)) * ((tp + fp) / (n + 1e-8)) +
          ((tn + fp) / (n + 1e-8)) * ((tn + fn) / (n + 1e-8)))
    return (po - pe) / (1.0 - pe + 1e-8)

def hausdorff95(gt, pred):
    gt = gt.astype(bool); pred = pred.astype(bool)
    gt_b   = morphology.binary_dilation(gt)   ^ gt
    pred_b = morphology.binary_dilation(pred) ^ pred
    if not gt_b.any() or not pred_b.any():
        return np.nan
    d_g2p = distance_transform_edt(~pred_b)[gt_b]
    d_p2g = distance_transform_edt(~gt_b)[pred_b]
    return float(np.percentile(np.hstack([d_g2p, d_p2g]), 95))

def normalized_surface_distance(gt, pred, delta=2.0):
    gt = gt.astype(bool); pred = pred.astype(bool)
    gt_b   = morphology.binary_dilation(gt)   ^ gt
    pred_b = morphology.binary_dilation(pred) ^ pred
    if not gt_b.any() or not pred_b.any():
        return np.nan
    d_g2p = distance_transform_edt(~pred_b)[gt_b]
    d_p2g = distance_transform_edt(~gt_b)[pred_b]
    within = np.sum(d_g2p <= delta) + np.sum(d_p2g <= delta)
    return within / (len(d_g2p) + len(d_p2g) + 1e-8)

def compute_all_metrics(gt, pred):
    gt   = (gt   > 0.5).astype(np.uint8)
    pred = (pred > 0.5).astype(np.uint8)
    return {
        "DSC"  : dice_score(gt, pred),
        "IoU"  : jaccard_index(gt, pred),
        "Sens" : sensitivity(gt, pred),
        "Spec" : specificity(gt, pred),
        "MCC"  : matthews_correlation_coefficient(gt, pred),
        "Kappa": cohens_kappa(gt, pred),
        "HD95" : hausdorff95(gt, pred),
        "NSD"  : normalized_surface_distance(gt, pred, delta=2.0),
    }


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────
def build_model():
    """
    DeepLabV3+ with ResNet-50 encoder.

    Architecture (matching paper):
      Encoder  : ResNet-50, pretrained on ImageNet
                 Uses Atrous (dilated) convolutions in the backbone
                 to preserve spatial resolution instead of striding.
      ASPP     : Atrous Spatial Pyramid Pooling — multi-scale context
                 with dilations [6, 12, 18] + global average pooling.
      Decoder  : Low-level feature skip from encoder layer1 (stride-4),
                 1×1 conv to reduce channels, concat with upsampled ASPP
                 output, 3×3 conv, bilinear upsample to input resolution.
      Output   : Single-channel logit map (sigmoid applied at inference).

    in_channels = 1  (grayscale X-rays)
    classes     = 1  (binary: lung vs background)
    """
    model = smp.DeepLabV3Plus(
        encoder_name    = "resnet50",
        encoder_weights = "imagenet",
        in_channels     = 1,
        classes         = 1,
        activation      = None,
    )
    return model.to(DEVICE)


# ──────────────────────────────────────────────────────────────
# CHECKPOINT UTILITIES
# ──────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scaler, epoch, best_val_dice, path):
    torch.save({
        "epoch"               : epoch,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict"   : scaler.state_dict() if scaler else None,
        "best_val_dice"       : best_val_dice,
    }, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    epoch         = ckpt.get("epoch", 0)
    best_val_dice = ckpt.get("best_val_dice", 0.0)
    print(f"  [ckpt] loaded from {path}  (epoch={epoch}, best_dice={best_val_dice:.4f})")
    return model, optimizer, scaler, epoch, best_val_dice


def load_weights_only(path, device=None):
    device = device or DEVICE
    model  = build_model()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"  [weights] loaded from {path}")
    return model


# ──────────────────────────────────────────────────────────────
# TRAIN / VALIDATE
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0

    for imgs, masks, *_ in tqdm(loader, desc="  Train", leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=(USE_AMP and DEVICE == "cuda")):
            outputs = model(imgs)
            loss    = criterion(outputs, masks)

        if USE_AMP and DEVICE == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    accs, dices, ious = [], [], []

    with torch.no_grad():
        for imgs, masks, *_ in tqdm(loader, desc="  Val  ", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs     = model(imgs)
            loss        = criterion(outputs, masks)
            total_loss += loss.item()
            acc, dice, iou = batch_metrics(outputs, masks)
            accs.append(acc); dices.append(dice); ious.append(iou)

    return (total_loss / len(loader),
            float(np.mean(accs)),
            float(np.mean(dices)),
            float(np.mean(ious)))


def save_visual_result(orig_img, pred_mask, save_path, gt_mask=None):

    # Convert tensor to numpy if needed
    if torch.is_tensor(orig_img):
        orig_img = orig_img.cpu().numpy()

    if orig_img.dtype != np.uint8:
        orig_img = orig_img.astype(np.uint8)

    if gt_mask is not None:
        if torch.is_tensor(gt_mask):
            gt_mask = gt_mask.cpu().numpy()

        gt = (gt_mask * 255).astype(np.uint8)

        canvas = np.concatenate(
            [orig_img, gt, pred_mask],
            axis=1
        )
    else:
        canvas = np.concatenate(
            [orig_img, pred_mask],
            axis=1
        )

    cv2.imwrite(save_path, canvas)

# ──────────────────────────────────────────────────────────────
# SAVE PREDICTIONS + FULL 8-METRIC EVALUATION
# ──────────────────────────────────────────────────────────────
def save_predictions(model, loader, save_dir, split_name, has_gt=True):
    """
    Saves predicted masks as PNG files and computes all 8 metrics per image.
    Writes a CSV to LOG_DIR/{split_name}_metrics.csv.
    """
    print(f"\n  Evaluating & saving predictions for [{split_name}] …")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    records = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {split_name}", leave=False):
            if has_gt:
                imgs, masks, orig_imgs, img_names, img_paths, mask_paths = batch
                masks = masks.to(DEVICE)
            else:
                imgs, orig_imgs, img_names, img_paths = batch

            imgs    = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.sigmoid(outputs)
            preds   = (probs > 0.5).float().cpu().numpy()

            for i in range(imgs.size(0)):
                name      = img_names[i]
                pred_mask = (preds[i,0] * 255).astype(np.uint8)
                orig = orig_imgs[i]

                if has_gt:
                    gt = masks[i,0].cpu().numpy()

                    save_visual_result(
                        orig,
                        pred_mask,
                        os.path.join(save_dir, name),
                        gt
                    )
                else:
                    save_visual_result(
                        orig,
                        pred_mask,
                        os.path.join(save_dir, name)
                    )

                if has_gt:
                    gt_np   = masks[i, 0].cpu().numpy()
                    pred_np = preds[i, 0]
                    m       = compute_all_metrics(gt_np, pred_np)
                    records.append({
                        "filename"  : name,
                        "image_path": img_paths[i],
                        "mask_path" : mask_paths[i],
                        **m,
                    })

    if records:
        df  = pd.DataFrame(records)
        csv = os.path.join(LOG_DIR, f"{split_name}_metrics.csv")
        df.to_csv(csv, index=False)

        num_cols = ["DSC", "IoU", "Sens", "Spec", "MCC", "Kappa", "HD95", "NSD"]
        print(f"\n  ── {split_name} summary ──")
        for col in num_cols:
            vals = df[col].dropna()
            print(f"    {col:<6}  mean={vals.mean():.4f}  std={vals.std():.4f}")
        print(f"    CSV  → {csv}")


# ──────────────────────────────────────────────────────────────
# INFERENCE HELPER  (standalone, post-training)
# ──────────────────────────────────────────────────────────────
def infer_single(image_path, weights_path, image_size=256, use_clahe=True,
                 threshold=0.5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_weights_only(weights_path, device)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h0, w0 = img.shape
    img = cv2.resize(img, (image_size, image_size))

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img   = clahe.apply(img)

    tensor = (torch.tensor(img / 255.0, dtype=torch.float32)
              .unsqueeze(0).unsqueeze(0).to(device))

    with torch.no_grad():
        logit    = model(tensor)
        prob_map = torch.sigmoid(logit)[0, 0].cpu().numpy()

    pred_mask = ((prob_map > threshold) * 255).astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    prob_map  = cv2.resize(prob_map,  (w0, h0), interpolation=cv2.INTER_LINEAR)
    return pred_mask, prob_map


# ──────────────────────────────────────────────────────────────
# MAIN  — guarded so DataLoader workers don't re-run it on Windows
# ──────────────────────────────────────────────────────────────
def main():
    seed_everything(SEED)
    clear_gpu()

    # Print device info exactly once, here inside main()
    print(f"Using device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM         : {total_mem:.1f} GB")

    print("\n" + "="*60)
    print("  DATA LOADING")
    print("="*60)

    # ── Collect paired (image, mask) paths from each split ──────
    train_imgs, train_masks = collect_split("train")
    val_imgs,   val_masks   = collect_split("val")
    test_imgs,  test_masks  = collect_split("test")

    # test/synth has CXRs only — no GT masks
    synth_imgs = collect_split_no_gt("test", "synth")

    print(f"  Train (all categories)  : {len(train_imgs)} paired samples")
    print(f"  Val   (all categories)  : {len(val_imgs)}   paired samples")
    print(f"  Test  (with GT)         : {len(test_imgs)}  paired samples")
    print(f"  Test  synth (no GT)     : {len(synth_imgs)} samples")

    if len(train_imgs) == 0:
        raise RuntimeError(
            "No training samples found. "
            "Check that Standard_Data/train/<category>/cxr/ and masks/ exist."
        )

    # ── Build datasets ───────────────────────────────────────────
    train_ds = LungDataset(train_imgs, train_masks, IMAGE_SIZE, USE_CLAHE)
    val_ds   = LungDataset(val_imgs,   val_masks,   IMAGE_SIZE, USE_CLAHE)
    test_ds  = LungDataset(test_imgs,  test_masks,  IMAGE_SIZE, USE_CLAHE)
    synth_ds = LungDataset(synth_imgs, None,        IMAGE_SIZE, USE_CLAHE)

    # ── Build loaders ────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)
    synth_loader = DataLoader(synth_ds, batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Used only for saving final qualitative results
    train_vis_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("\n" + "="*60)
    print("  MODEL")
    print("="*60)
    model     = build_model()
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # StepLR: halve LR every 5 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")
    print(f"  Early stopping   : patience={EARLY_STOP_PATIENCE} epochs (val Dice)")

    best_val_dice    = 0.0
    best_model_wts   = copy.deepcopy(model.state_dict())
    epoch_records    = []
    no_improve_count = 0          # early-stopping counter

    print("\n" + "="*60)
    print("  TRAINING  (full train set → full val set each epoch)")
    print("="*60)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}  (LR={scheduler.get_last_lr()[0]:.6f})")

        # ── Full training pass ───────────────────────────────────
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)

        # ── Full validation pass ─────────────────────────────────
        val_loss, val_acc, val_dice, val_iou = validate(model, val_loader, criterion)

        scheduler.step()

        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}  |  Acc: {val_acc:.4f}  |  "
              f"Dice: {val_dice:.4f}  |  IoU: {val_iou:.4f}")

        epoch_records.append({
            "Epoch"            : epoch + 1,
            "Train Loss"       : train_loss,
            "Val Loss"         : val_loss,
            "Val Accuracy"     : val_acc,
            "Val Dice (DSC)"   : val_dice,
            "Val IoU (Jaccard)": val_iou,
        })

        # Always overwrite the latest checkpoint
        save_checkpoint(model, optimizer, scaler, epoch + 1, best_val_dice,
                        os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth"))

        # Periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scaler, epoch + 1, best_val_dice,
                            os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

        # ── Best-model tracking + early stopping ────────────────
        if val_dice > best_val_dice:
            best_val_dice    = val_dice
            best_model_wts   = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            save_checkpoint(model, optimizer, scaler, epoch + 1, best_val_dice,
                            os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth"))
            torch.save(model.state_dict(),
                       os.path.join(WEIGHTS_DIR, "best_model_weights.pth"))
            print(f"  ★ New best Val Dice = {best_val_dice:.4f}")
        else:
            no_improve_count += 1
            print(f"  No improvement for {no_improve_count}/{EARLY_STOP_PATIENCE} epoch(s)")
            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"\n  [Early Stop] Val Dice did not improve for "
                      f"{EARLY_STOP_PATIENCE} consecutive epochs. Stopping training.")
                break

    # ── Final weights (last epoch state, regardless of early stop) ──
    torch.save(model.state_dict(),
               os.path.join(WEIGHTS_DIR, "final_model_weights.pth"))

    pd.DataFrame(epoch_records).to_csv(
        os.path.join(LOG_DIR, "epoch_metrics.csv"), index=False)

    # ── Restore best weights before testing ─────────────────────
    model.load_state_dict(best_model_wts)
    print(f"\n  Restored best weights (Val Dice = {best_val_dice:.4f}) for testing.")

    print("\n" + "="*60)
    print("  TESTING  (8 metrics per image, runs once after training)")
    print("="*60)

    print("\nSaving qualitative predictions...")

    save_predictions(
        model,
        train_vis_loader,
        TRAIN_PRED_DIR,
        "train",
        has_gt=True
    )

    save_predictions(
        model,
        val_loader,
        VAL_PRED_DIR,
        "val",
        has_gt=True
    )

    save_predictions(
        model,
        test_loader,
        TEST_PRED_DIR,
        "test",
        has_gt=True
    )

    if len(synth_imgs) > 0:
        save_predictions(
            model,
            synth_loader,
            SYNTH_PRED_DIR,
            "test_synth",
            has_gt=False
        )

    print("\n" + "="*60)
    print("  DONE")
    print("="*60)
    print(f"  Best weights  → {os.path.join(WEIGHTS_DIR, 'best_model_weights.pth')}")
    print(f"  Final weights → {os.path.join(WEIGHTS_DIR, 'final_model_weights.pth')}")
    print(f"  Predictions   → {PRED_DIR}/")
    print(f"  Logs & CSVs   → {LOG_DIR}/")


if __name__ == "__main__":
    main()