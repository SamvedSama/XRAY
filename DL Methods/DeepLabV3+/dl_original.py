import os
import cv2
import gc
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp

# ======================================================
# CONFIGURATION
# ======================================================

SEED = 42
ROOT = "./Normalized Data"
COVIDQU_PATH = os.path.join(ROOT, "covidqu", "Lung Segmentation Data")
JSRT_PATH = os.path.join(ROOT, "jsrt")
SYNTH_PATH = os.path.join(ROOT, "synth", "cxr")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 15
LR = 1e-3
BATCH_SIZE = 2
IMAGE_SIZE = 256
USE_CLAHE = True

OUTPUT_DIR = "outputs_single"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "weights")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
TRAIN_PRED_DIR = os.path.join(PRED_DIR, "train")
VAL_PRED_DIR = os.path.join(PRED_DIR, "val")
TEST_PRED_DIR = os.path.join(PRED_DIR, "test")
SYNTH_PRED_DIR = os.path.join(PRED_DIR, "synth")

for d in [
    OUTPUT_DIR, CHECKPOINT_DIR, WEIGHTS_DIR, LOG_DIR,
    PRED_DIR, TRAIN_PRED_DIR, VAL_PRED_DIR, TEST_PRED_DIR, SYNTH_PRED_DIR
]:
    os.makedirs(d, exist_ok=True)

# ======================================================
# REPRODUCIBILITY
# ======================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# ======================================================
# GPU MEMORY CLEANUP
# ======================================================

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

clear_gpu()
print(f"Using device: {DEVICE}")

# ======================================================
# DATA COLLECTION
# ======================================================

def collect_covidqu(split):
    imgs, masks = [], []
    split_path = os.path.join(COVIDQU_PATH, split)

    for category in ["Normal", "COVID-19", "Non-COVID"]:
        base = os.path.join(split_path, category)
        img_dir = os.path.join(base, "images")
        mask_dir = os.path.join(base, "lung masks")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            continue

        for file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, file)
            mask_path = os.path.join(mask_dir, file)

            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                imgs.append(img_path)
                masks.append(mask_path)

    return imgs, masks


def collect_jsrt():
    img_dir = os.path.join(JSRT_PATH, "cxr")
    mask_dir = os.path.join(JSRT_PATH, "masks")

    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)]) if os.path.exists(img_dir) else []
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]) if os.path.exists(mask_dir) else []

    return imgs, masks


def collect_synth():
    if not os.path.exists(SYNTH_PATH):
        return []
    return sorted([os.path.join(SYNTH_PATH, f) for f in os.listdir(SYNTH_PATH)])


# ======================================================
# DATASET
# ======================================================

class LungDataset(Dataset):
    def __init__(self, images, masks=None, image_size=256, use_clahe=False):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_name = os.path.basename(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.image_size, self.image_size))

        if self.use_clahe:
            clahe = cv2.createCLAHE(2.0, (8, 8))
            img = clahe.apply(img)

        img_tensor = torch.tensor(img / 255.0).unsqueeze(0).float()

        if self.masks is not None:
            mask_path = self.masks[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask = (mask > 127).astype("float32")
            mask_tensor = torch.tensor(mask).unsqueeze(0).float()

            return img_tensor, mask_tensor, img, img_name, img_path, mask_path
        else:
            return img_tensor, img, img_name, img_path


# ======================================================
# METRICS
# ======================================================

def compute_metrics(pred, mask):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    TP = (pred * mask).sum()
    TN = ((1 - pred) * (1 - mask)).sum()
    FP = (pred * (1 - mask)).sum()
    FN = ((1 - pred) * mask).sum()

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)

    return acc.item(), dice.item(), iou.item()


# ======================================================
# LOSS
# ======================================================

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred = torch.sigmoid(pred)

        smooth = 1e-8
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice = (2. * intersection + smooth) / (
            pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth
        )
        dice_loss = 1 - dice.mean()

        return bce + dice_loss


# ======================================================
# MODEL
# ======================================================

def build_model():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None
    )
    return model.to(DEVICE)


# ======================================================
# SAVE / LOAD UTILITIES
# ======================================================

def save_checkpoint(model, optimizer, epoch, best_val_dice, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice
    }, path)


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_val_dice = checkpoint.get("best_val_dice", 0.0)

    return model, optimizer, epoch, best_val_dice


def load_weights_only(path):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# ======================================================
# SAVE PREDICTIONS
# ======================================================

def save_predictions(model, loader, save_dir, split_name, has_gt=True):
    print(f"\nSaving predictions for {split_name}...")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    records = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Saving {split_name}"):

            if has_gt:
                imgs, masks, original_imgs, img_names, img_paths, mask_paths = batch
                masks = masks.to(DEVICE)
            else:
                imgs, original_imgs, img_names, img_paths = batch

            imgs = imgs.to(DEVICE)
            outputs = model(imgs)

            pred = torch.sigmoid(outputs)
            pred = (pred > 0.5).float().cpu().numpy()

            for i in range(imgs.size(0)):
                img_name = img_names[i]
                pred_mask = (pred[i, 0] * 255).astype(np.uint8)

                if has_gt:
                    acc, dice, iou = compute_metrics(
                        outputs[i:i+1],
                        masks[i:i+1]
                    )

                    records.append({
                        "filename": img_name,
                        "image_path": img_paths[i],
                        "mask_path": mask_paths[i],
                        "Global Accuracy": acc,
                        "Dice (DSC)": dice,
                        "IoU (Jaccard)": iou
                    })

                # Save ONLY predicted mask
                cv2.imwrite(os.path.join(save_dir, img_name), pred_mask)

                if has_gt and len(records) > 0:
                    pd.DataFrame(records).to_csv(
                        os.path.join(LOG_DIR, f"{split_name}_metrics.csv"),
                        index=False
                    )


# ======================================================
# TRAIN / VALIDATE
# ======================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_loss = 0

    for imgs, masks, _, _, _, _ in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate(model, loader):
    model.eval()
    accs, dices, ious = [], [], []
    val_loss = 0.0

    criterion = DiceBCELoss()

    with torch.no_grad():
        for imgs, masks, _, _, _, _ in tqdm(loader, desc="Validation"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            acc, dice, iou = compute_metrics(outputs, masks)
            accs.append(acc)
            dices.append(dice)
            ious.append(iou)

    return (
        val_loss / len(loader),
        np.mean(accs),
        np.mean(dices),
        np.mean(ious)
    )


# ======================================================
# MAIN
# ======================================================

def main():
    print("\n==================== DATA LOADING ====================")

    train_imgs, train_masks = collect_covidqu("Train")
    test_imgs, test_masks = collect_covidqu("Test")
    jsrt_imgs, jsrt_masks = collect_jsrt()
    synth_imgs = collect_synth()

    print(f"CovidQU Train samples: {len(train_imgs)}")
    print(f"CovidQU Test samples : {len(test_imgs)}")
    print(f"JSRT samples         : {len(jsrt_imgs)}")
    print(f"Synth samples        : {len(synth_imgs)}")

    # Split train into train + val
    full_train_dataset = LungDataset(train_imgs, train_masks, IMAGE_SIZE, USE_CLAHE)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    test_dataset = LungDataset(test_imgs, test_masks, IMAGE_SIZE, USE_CLAHE)
    jsrt_dataset = LungDataset(jsrt_imgs, jsrt_masks, IMAGE_SIZE, USE_CLAHE)
    synth_dataset = LungDataset(synth_imgs, None, IMAGE_SIZE, USE_CLAHE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    jsrt_loader = DataLoader(jsrt_dataset, batch_size=1, shuffle=False)
    synth_loader = DataLoader(synth_dataset, batch_size=1, shuffle=False)

    print("\n==================== MODEL ====================")
    model = build_model()
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_dice = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epoch_records = []

    print("\n==================== TRAINING ====================")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_dice, val_iou = validate(model, val_loader)

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        epoch_records.append({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Val Global Accuracy": val_acc,
            "Val Dice (DSC)": val_dice,
            "Val IoU (Jaccard)": val_iou
        })

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch + 1, best_val_dice,
            os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
        )

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, best_val_dice,
                os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            )

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_model_wts = copy.deepcopy(model.state_dict())

            save_checkpoint(
                model, optimizer, epoch + 1, best_val_dice,
                os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")
            )

            torch.save(
                model.state_dict(),
                os.path.join(WEIGHTS_DIR, "best_model_weights.pth")
            )

    # Save final model weights
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "final_model_weights.pth"))

    # Load best model for evaluation
    model.load_state_dict(best_model_wts)

    pd.DataFrame(epoch_records).to_csv(
        os.path.join(LOG_DIR, "epoch_metrics.csv"),
        index=False
    )

    print("\n==================== SAVING OUTPUTS ====================")

    save_predictions(model, train_loader, TRAIN_PRED_DIR, "train", has_gt=True)
    save_predictions(model, val_loader, VAL_PRED_DIR, "val", has_gt=True)
    save_predictions(model, test_loader, TEST_PRED_DIR, "test", has_gt=True)
    save_predictions(model, jsrt_loader, os.path.join(PRED_DIR, "jsrt"), "jsrt", has_gt=True)
    save_predictions(model, synth_loader, SYNTH_PRED_DIR, "synth", has_gt=False)

    print("\n==================== DONE ====================")
    print(f"Best model weights saved to: {os.path.join(WEIGHTS_DIR, 'best_model_weights.pth')}")
    print(f"Final model weights saved to: {os.path.join(WEIGHTS_DIR, 'final_model_weights.pth')}")
    print(f"Predictions saved in: {PRED_DIR}")


if __name__ == "__main__":
    main()