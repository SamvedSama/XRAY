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
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# ======================================================
# CONFIGURATION
# ======================================================

SEED = 42
ROOT = os.path.join("..", "DL Methods", "DeepLabV3+", "Normalized Data")
COVIDQU_PATH = os.path.join(ROOT, "covidqu", "Lung Segmentation Data")
JSRT_PATH = os.path.join(ROOT, "jsrt")
SYNTH_PATH = os.path.join(ROOT, "synth", "cxr")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
EPOCHS = 50
LR = 1e-5
BATCH_SIZE = 2
USE_AUGMENTATION = True
USE_CLAHE = False

OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "weights")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
TRAIN_PRED_DIR = os.path.join(PRED_DIR, "train")
VAL_PRED_DIR = os.path.join(PRED_DIR, "val")
TEST_PRED_DIR = os.path.join(PRED_DIR, "test")
JSRT_PRED_DIR = os.path.join(PRED_DIR, "jsrt")
SYNTH_PRED_DIR = os.path.join(PRED_DIR, "synth")

for d in [
    OUTPUT_DIR, CHECKPOINT_DIR, WEIGHTS_DIR, LOG_DIR,
    PRED_DIR, TRAIN_PRED_DIR, VAL_PRED_DIR,
    TEST_PRED_DIR, JSRT_PRED_DIR, SYNTH_PRED_DIR
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
            print(f"Warning: Missing folder -> {img_dir} or {mask_dir}")
            continue

        for file in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, file)
            mask_path = os.path.join(mask_dir, file)

            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                imgs.append(img_path)
                masks.append(mask_path)

    return imgs, masks


def collect_jsrt():
    imgs, masks = [], []

    img_dir = os.path.join(JSRT_PATH, "cxr")
    mask_dir = os.path.join(JSRT_PATH, "masks")

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"Warning: Missing JSRT folder -> {img_dir} or {mask_dir}")
        return imgs, masks

    for file in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, file)
        mask_path = os.path.join(mask_dir, file)

        if os.path.isfile(img_path) and os.path.isfile(mask_path):
            imgs.append(img_path)
            masks.append(mask_path)

    return imgs, masks


def collect_synth():
    imgs = []

    if not os.path.exists(SYNTH_PATH):
        print(f"Warning: Missing SYNTH folder -> {SYNTH_PATH}")
        return imgs

    for file in sorted(os.listdir(SYNTH_PATH)):
        img_path = os.path.join(SYNTH_PATH, file)
        if os.path.isfile(img_path):
            imgs.append(img_path)

    return imgs

# ======================================================
# DATASET
# ======================================================

class LungDataset(Dataset):
    def __init__(self, images, masks=None, image_size=224, use_augmentation=False, use_clahe=False):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.images)

    def augment_image(self, img):
        img = img.astype(np.float32) / 255.0

        # Random brightness
        delta = np.random.uniform(-0.2, 0.2)
        img = img + delta

        # Random contrast
        contrast = np.random.uniform(0.2, 0.8)
        mean = img.mean()
        img = (img - mean) * contrast + mean

        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_name = os.path.basename(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        img = cv2.resize(img, (self.image_size, self.image_size))

        if self.use_clahe:
            clahe = cv2.createCLAHE(2.0, (8, 8))
            img = clahe.apply(img)

        if self.use_augmentation:
            img = self.augment_image(img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_tensor = torch.tensor(img_rgb / 255.0).permute(2, 0, 1).float()

        if self.masks is not None:
            mask_path = self.masks[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")

            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask = (mask > 127).astype(np.float32)
            mask_tensor = torch.tensor(mask).unsqueeze(0).float()

            return img_tensor, mask_tensor, img, img_name, img_path, mask_path
        else:
            return img_tensor, img, img_name, img_path

# ======================================================
# METRICS
# ======================================================

def dice_coefficient(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return dice.mean().item()


def iou_score(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def pixel_accuracy(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()

# ======================================================
# LOSS
# ======================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        )

        return 1.0 - dice.mean()

# ======================================================
# PAPER MODEL BLOCKS
# ======================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation if kernel_size == 3 else 0
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.3):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dropout = nn.Dropout2d(dropout)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

# ======================================================
# PAPER DDRU-NET (DenseNet201 Encoder)
# ======================================================

class DDRUNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        features = backbone.features

        # DenseNet201 encoder parts
        self.stem = nn.Sequential(
            features.conv0,
            features.norm0,
            features.relu0
        )  # 64 channels, 112x112

        self.pool0 = features.pool0

        self.denseblock1 = features.denseblock1
        self.transition1 = features.transition1

        self.denseblock2 = features.denseblock2
        self.transition2 = features.transition2

        self.denseblock3 = features.denseblock3
        self.transition3 = features.transition3

        self.denseblock4 = features.denseblock4
        self.norm5 = features.norm5

        # Bottleneck with dilated convolutions
        self.bottleneck = nn.Sequential(
            ConvBNReLU(1920, 1024, kernel_size=3, dilation=2),
            ConvBNReLU(1024, 1024, kernel_size=3, dilation=2)
        )

        # Decoder (correct DenseNet201 skip sizes)
        self.dec1 = DecoderBlock(1024, 1792, 512)
        self.dec2 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec4 = DecoderBlock(128, 64, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        s1 = self.stem(x)               # [B, 64, 112, 112]
        x = self.pool0(s1)              # [B, 64, 56, 56]

        x = self.denseblock1(x)
        s2 = x                          # [B, 256, 56, 56]
        x = self.transition1(x)         # [B, 128, 28, 28]

        x = self.denseblock2(x)
        s3 = x                          # [B, 512, 28, 28]
        x = self.transition2(x)         # [B, 256, 14, 14]

        x = self.denseblock3(x)
        s4 = x                          # [B, 1792, 14, 14]
        x = self.transition3(x)         # [B, 896, 7, 7]

        x = self.denseblock4(x)
        x = self.norm5(x)               # [B, 1920, 7, 7]

        # Bottleneck
        b = self.bottleneck(x)

        # Decoder
        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)

        out = self.final_up(d4)
        out = self.final_conv(out)

        return out

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
    model = DDRUNet().to(DEVICE)
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
                orig = original_imgs[i].numpy() if isinstance(original_imgs[i], torch.Tensor) else original_imgs[i]
                pred_mask = (pred[i, 0] * 255).astype(np.uint8)

                if has_gt:
                    gt = (masks[i].cpu().numpy()[0] * 255).astype(np.uint8)
                    combined = np.hstack([orig.astype(np.uint8), gt, pred_mask])

                    acc = pixel_accuracy(outputs[i:i+1], masks[i:i+1])
                    dice = dice_coefficient(outputs[i:i+1], masks[i:i+1])
                    iou = iou_score(outputs[i:i+1], masks[i:i+1])

                    records.append({
                        "filename": img_name,
                        "image_path": img_paths[i],
                        "mask_path": mask_paths[i],
                        "Global Accuracy": acc,
                        "Dice (DSC)": dice,
                        "IoU (Jaccard)": iou
                    })
                else:
                    combined = np.hstack([orig.astype(np.uint8), pred_mask])

                cv2.imwrite(os.path.join(save_dir, img_name), combined)

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


def validate(model, loader, criterion):
    model.eval()
    losses, accs, dices, ious = [], [], [], []

    with torch.no_grad():
        for imgs, masks, _, _, _, _ in tqdm(loader, desc="Validation"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)

            loss = criterion(outputs, masks)
            losses.append(loss.item())

            accs.append(pixel_accuracy(outputs, masks))
            dices.append(dice_coefficient(outputs, masks))
            ious.append(iou_score(outputs, masks))

    return np.mean(losses), np.mean(accs), np.mean(dices), np.mean(ious)

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

    if len(train_imgs) == 0:
        raise ValueError("No training images found. Check dataset paths.")

    # ==================================================
    # SAFE TRAIN/VAL SPLIT (NO AUGMENTATION LEAK)
    # ==================================================
    indices = list(range(len(train_imgs)))
    random.seed(SEED)
    random.shuffle(indices)

    train_size = int(0.8 * len(indices))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_split_imgs = [train_imgs[i] for i in train_idx]
    train_split_masks = [train_masks[i] for i in train_idx]

    val_split_imgs = [train_imgs[i] for i in val_idx]
    val_split_masks = [train_masks[i] for i in val_idx]

    train_dataset = LungDataset(
        train_split_imgs,
        train_split_masks,
        image_size=IMG_SIZE,
        use_augmentation=USE_AUGMENTATION,
        use_clahe=USE_CLAHE
    )

    val_dataset = LungDataset(
        val_split_imgs,
        val_split_masks,
        image_size=IMG_SIZE,
        use_augmentation=False,
        use_clahe=USE_CLAHE
    )

    test_dataset = LungDataset(
        test_imgs, test_masks,
        image_size=IMG_SIZE,
        use_augmentation=False,
        use_clahe=USE_CLAHE
    )

    jsrt_dataset = LungDataset(
        jsrt_imgs, jsrt_masks,
        image_size=IMG_SIZE,
        use_augmentation=False,
        use_clahe=USE_CLAHE
    )

    synth_dataset = LungDataset(
        synth_imgs, None,
        image_size=IMG_SIZE,
        use_augmentation=False,
        use_clahe=USE_CLAHE
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    jsrt_loader = DataLoader(jsrt_dataset, batch_size=1, shuffle=False, num_workers=0)
    synth_loader = DataLoader(synth_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("\n==================== MODEL ====================")
    model = DDRUNet().to(DEVICE)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10
    )

    best_val_dice = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    patience_counter = 0
    epoch_records = []
    early_stop_patience = 10
    min_delta = 0.001

    print("\n==================== TRAINING ====================")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_dice, val_iou = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

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
            "Val IoU (Jaccard)": val_iou,
            "Learning Rate": optimizer.param_groups[0]["lr"]
        })

        # Save latest checkpoint every epoch
        save_checkpoint(
            model, optimizer, epoch + 1, best_val_dice,
            os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, best_val_dice,
                os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            )

        # ==================================================
        # BEST MODEL + EARLY STOPPING (DICE BASED)
        # ==================================================
        if val_dice > best_val_dice + min_delta:
            best_val_dice = val_dice
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0

            print(f"New best Dice: {best_val_dice:.4f} -> saving model")

            save_checkpoint(
                model, optimizer, epoch + 1, best_val_dice,
                os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")
            )

            torch.save(
                model.state_dict(),
                os.path.join(WEIGHTS_DIR, "best_model_weights.pth")
            )
        else:
            patience_counter += 1
            print(f"No significant Dice improvement. Patience: {patience_counter}/{early_stop_patience}")

        if patience_counter >= early_stop_patience:
            print("\nEarly stopping triggered based on Dice.")
            break

    # Save final weights
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "final_model_weights.pth"))

    # Load best model for final evaluation
    model.load_state_dict(best_model_wts)

    pd.DataFrame(epoch_records).to_csv(
        os.path.join(LOG_DIR, "epoch_metrics.csv"),
        index=False
    )

    print("\n==================== SAVING OUTPUTS ====================")

    # WARNING: This can save many files if your train set is huge.
    save_predictions(model, train_loader, TRAIN_PRED_DIR, "train", has_gt=True)
    save_predictions(model, val_loader, VAL_PRED_DIR, "val", has_gt=True)
    save_predictions(model, test_loader, TEST_PRED_DIR, "test", has_gt=True)
    save_predictions(model, jsrt_loader, JSRT_PRED_DIR, "jsrt", has_gt=True)
    save_predictions(model, synth_loader, SYNTH_PRED_DIR, "synth", has_gt=False)

    print("\n==================== DONE ====================")
    print(f"Best model weights saved to: {os.path.join(WEIGHTS_DIR, 'best_model_weights.pth')}")
    print(f"Final model weights saved to: {os.path.join(WEIGHTS_DIR, 'final_model_weights.pth')}")
    print(f"Predictions saved in: {PRED_DIR}")


if __name__ == "__main__":
    main()