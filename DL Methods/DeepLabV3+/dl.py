import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import pandas as pd

# ======================================================
# CONFIGURATION
# ======================================================

ROOT = "Normalized Data"
COVIDQU_PATH = os.path.join(ROOT, "covidqu", "Lung Segmentation Data")
JSRT_PATH = os.path.join(ROOT, "jsrt")
SYNTH_PATH = os.path.join(ROOT, "synth", "cxr")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 15
LR = 0.001
BATCH_SIZE = 2

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

        for file in os.listdir(img_dir):
            imgs.append(os.path.join(img_dir, file))
            masks.append(os.path.join(mask_dir, file))

    return imgs, masks


def collect_jsrt():
    img_dir = os.path.join(JSRT_PATH, "cxr")
    mask_dir = os.path.join(JSRT_PATH, "masks")

    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    return imgs, masks


def collect_synth():
    return [os.path.join(SYNTH_PATH, f) for f in os.listdir(SYNTH_PATH)]

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
        img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.image_size, self.image_size))

        if self.use_clahe:
            clahe = cv2.createCLAHE(2.0, (8, 8))
            img = clahe.apply(img)

        img_tensor = torch.tensor(img / 255.0).unsqueeze(0).float()

        if self.masks:
            mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask = (mask > 127).astype("float32")
            mask_tensor = torch.tensor(mask).unsqueeze(0).float()
            return img_tensor, mask_tensor, img
        else:
            return img_tensor, img

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
# EXPERIMENT FUNCTION
# ======================================================

def run_experiment(APPROACH):

    print("\n===================================================")
    print(f"Starting {APPROACH.upper()} Experiment")
    print("===================================================")

    if APPROACH == "paper":
        IMAGE_SIZE = 256
        USE_CLAHE = False
    else:
        IMAGE_SIZE = 512
        USE_CLAHE = True

    SAVE_DIR = f"results1_{APPROACH}"
    VIS_DIR = os.path.join(SAVE_DIR, "visual_results")

    os.makedirs(VIS_DIR, exist_ok=True)

    # Load datasets
    train_imgs, train_masks = collect_covidqu("Train")
    val_imgs, val_masks = collect_covidqu("Test")
    jsrt_imgs, jsrt_masks = collect_jsrt()
    synth_imgs = collect_synth()

    train_loader = DataLoader(
        LungDataset(train_imgs, train_masks, IMAGE_SIZE, USE_CLAHE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        LungDataset(val_imgs, val_masks, IMAGE_SIZE, USE_CLAHE),
        batch_size=1
    )

    jsrt_loader = DataLoader(
        LungDataset(jsrt_imgs, jsrt_masks, IMAGE_SIZE, USE_CLAHE),
        batch_size=1
    )

    synth_loader = DataLoader(
        LungDataset(synth_imgs, image_size=IMAGE_SIZE, use_clahe=USE_CLAHE),
        batch_size=1
    )

    # Model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)

    epoch_records = []

    # ================= TRAINING =================

    for epoch in range(EPOCHS):

        model.train()
        epoch_loss = 0

        for imgs, masks, _ in tqdm(train_loader, desc=f"{APPROACH} Training"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        accs, dices, ious = [], [], []

        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)
                acc, dice, iou = compute_metrics(outputs, masks)
                accs.append(acc)
                dices.append(dice)
                ious.append(iou)

        val_acc = np.mean(accs)
        val_dice = np.mean(dices)
        val_iou = np.mean(ious)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"Val IoU: {val_iou:.4f}")

        epoch_records.append({
            "Epoch": epoch+1,
            "Train Loss": avg_loss,
            "Val Global Accuracy": val_acc,
            "Val Dice (DSC)": val_dice,
            "Val IoU (Jaccard)": val_iou
        })

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

    # Save epoch metrics CSV
    pd.DataFrame(epoch_records).to_csv(
        os.path.join(SAVE_DIR, "epoch_metrics.csv"),
        index=False
    )

    # ================= TESTING =================

    def evaluate(loader, name):

        print(f"\nTesting on {name}")
        dataset_dir = os.path.join(VIS_DIR, name)
        os.makedirs(dataset_dir, exist_ok=True)

        image_records = []
        accs, dices, ious = [], [], []

        model.eval()

        with torch.no_grad():
            for idx, (imgs, masks, original_img) in enumerate(tqdm(loader)):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)

                acc, dice, iou = compute_metrics(outputs, masks)

                accs.append(acc)
                dices.append(dice)
                ious.append(iou)

                image_records.append({
                    "Image ID": idx,
                    "Global Accuracy": acc,
                    "Dice (DSC)": dice,
                    "IoU (Jaccard)": iou
                })

                pred = torch.sigmoid(outputs)
                pred = (pred > 0.5).float().cpu().numpy()[0,0] * 255
                gt = masks.cpu().numpy()[0,0] * 255
                orig = original_img[0]

                combined = np.hstack([orig, gt, pred.astype(np.uint8)])
                cv2.imwrite(os.path.join(dataset_dir, f"{idx}.png"), combined)

        pd.DataFrame(image_records).to_csv(
            os.path.join(dataset_dir, "per_image_metrics.csv"),
            index=False
        )

        return {
            "Global Accuracy": float(np.mean(accs)),
            "Dice (DSC)": float(np.mean(dices)),
            "IoU (Jaccard)": float(np.mean(ious))
        }

    def evaluate_synth(loader):

        print("\nTesting on SYNTH (No GT)")
        dataset_dir = os.path.join(VIS_DIR, "synth_test")
        os.makedirs(dataset_dir, exist_ok=True)

        model.eval()

        with torch.no_grad():
            for idx, (imgs, original_img) in enumerate(tqdm(loader)):
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)

                pred = torch.sigmoid(outputs)
                pred = (pred > 0.5).float().cpu().numpy()[0,0] * 255
                orig = original_img[0]

                combined = np.hstack([orig, pred.astype(np.uint8)])
                cv2.imwrite(os.path.join(dataset_dir, f"{idx}.png"), combined)

    evaluate(val_loader, "covidqu_test")
    evaluate(jsrt_loader, "jsrt_test")
    evaluate_synth(synth_loader)

    print(f"\nFinished {APPROACH.upper()} Experiment")

# ======================================================
# RUN BOTH
# ======================================================

run_experiment("paper")
run_experiment("improved")

print("\nAll Experiments Completed.")

# ======================================================
# END
# ======================================================