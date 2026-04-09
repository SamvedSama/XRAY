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

# ======================================================
# CONFIGURATION
# ======================================================

SEED = 42
ROOT = os.path.join("..", "..", "DL Methods", "DeepLabV3+", "Normalized Data")
COVIDQU_PATH = os.path.join(ROOT, "covidqu", "Lung Segmentation Data")
JSRT_PATH = os.path.join(ROOT, "jsrt")
SYNTH_PATH = os.path.join(ROOT, "synth", "cxr")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# ORIGINAL U-NET SPIRIT SETTINGS
# -----------------------------
IMG_SIZE = 256              # Practical chest X-ray setting
SAVE_MASK_SIZE = 256
EPOCHS = 100
LR = 0.01                    # SGD-friendly
BATCH_SIZE = 2               # Original U-Net spirit
MOMENTUM = 0.9              # Original paper
WEIGHT_DECAY = 0.0
USE_AUGMENTATION = True
USE_CLAHE = True
EARLY_STOP_PATIENCE = 10
MIN_DELTA = 0.001

# Optional bottleneck dropout (paper spirit)
DROPOUT_P = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_unet2015_fixed")
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

print("BASE_DIR                 :", BASE_DIR)
print("OUTPUT_DIR ABSOLUTE PATH :", OUTPUT_DIR)
print("Directories created successfully.")

# ======================================================
# REPRODUCIBILITY
# ======================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

clear_gpu()
print(f"Using device: {DEVICE}")

# ======================================================
# DATA COLLECTION (SAME STYLE AS YOUR CODE)
# ======================================================

def collect_covidqu(split):
    imgs, masks = [], []
    split_path = os.path.join(COVIDQU_PATH, split)

    for category in ["Normal", "Non-COVID"]:
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
    def __init__(self, images, masks=None, image_size=512, use_augmentation=False, use_clahe=False):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.images)

    def elastic_transform(self, image, mask, alpha=20, sigma=6):
        """Paper-inspired elastic deformation"""
        shape = image.shape
        random_state = np.random.RandomState(None)

        dx = cv2.GaussianBlur(
            (random_state.rand(*shape) * 2 - 1).astype(np.float32),
            (17, 17), sigma
        ) * alpha
        dy = cv2.GaussianBlur(
            (random_state.rand(*shape) * 2 - 1).astype(np.float32),
            (17, 17), sigma
        ) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if mask is not None:
            mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

        return image, mask

    def augment_image_and_mask(self, img, mask=None):
        # Horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)

        # Small rotation
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

        # Brightness / contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-15, 15)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # Elastic deformation
        if random.random() < 0.3:
            img, mask = self.elastic_transform(img, mask)

        return img, mask

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_name = os.path.basename(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

        mask = None
        mask_path = None

        if self.masks is not None:
            mask_path = self.masks[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")

            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)

        if self.use_augmentation:
            img, mask = self.augment_image_and_mask(img, mask)

        img_tensor = torch.tensor(img / 255.0).unsqueeze(0).float()

        if mask is not None:
            mask_tensor = torch.tensor(mask).unsqueeze(0).float()
            return img_tensor, mask_tensor, img_name, img_path, mask_path, (orig_h, orig_w)
        else:
            return img_tensor, img_name, img_path, (orig_h, orig_w)

# ======================================================
# METRICS
# ======================================================

def dice_from_probs(pred_bin, target, smooth=1.0):
    pred_bin = pred_bin.view(pred_bin.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred_bin * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred_bin.sum(dim=1) + target.sum(dim=1) + smooth)
    return dice.mean().item()

def iou_from_probs(pred_bin, target, smooth=1.0):
    pred_bin = pred_bin.view(pred_bin.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred_bin * target).sum(dim=1)
    union = pred_bin.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def get_binary_pred(logits):
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    return pred

def dice_coefficient(pred_logits, target, smooth=1.0):
    pred = get_binary_pred(pred_logits)
    return dice_from_probs(pred, target, smooth)

def iou_score(pred_logits, target, smooth=1.0):
    pred = get_binary_pred(pred_logits)
    return iou_from_probs(pred, target, smooth)

# ======================================================
# LOSS
# ======================================================

class DiceBCELoss(nn.Module):
    """
    Practical chest X-ray loss while still being a very strong U-Net baseline.
    Better than raw BCE alone for medical masks.
    """
    def __init__(self, smooth=1.0, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce = self.bce(pred, target)

        probs = torch.sigmoid(pred)
        probs = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (probs * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss

# ======================================================
# ORIGINAL U-NET 2015 STYLE MODEL
# ======================================================

class DoubleConv(nn.Module):
    """
    Original U-Net spirit:
    - 2 x 3x3 conv
    - ReLU
    - NO BatchNorm
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2015(nn.Module):
    """
    Faithful in architecture spirit to the original 2015 U-Net:
    - 64,128,256,512,1024 channels
    - 2 convs per stage
    - maxpool downsampling
    - transpose conv upsampling
    - skip connections
    - dropout near bottleneck
    """
    def __init__(self, in_channels=1, out_channels=1, dropout_p=0.5):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)
        self.dropout = nn.Dropout2d(dropout_p)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """
        He/Kaiming init = faithful to original ReLU-network initialization intent
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)
        b = self.dropout(b)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
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

# ======================================================
# IMAGE-WISE EVALUATION + PREDICTION SAVING
# ======================================================

def evaluate_and_save_predictions(model, loader, save_dir, split_name, has_gt=True):
    print(f"\nSaving predictions for {split_name}...")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    image_records = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Saving {split_name}"):

            if has_gt:
                imgs, masks, img_names, img_paths, mask_paths, orig_sizes = batch
                masks = masks.to(DEVICE)
            else:
                imgs, img_names, img_paths, orig_sizes = batch

            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            pred = get_binary_pred(outputs).cpu().numpy()

            for i in range(imgs.size(0)):
                img_name = img_names[i]
                orig_h = int(orig_sizes[0][i].item()) if isinstance(orig_sizes[0], torch.Tensor) else int(orig_sizes[i][0])
                orig_w = int(orig_sizes[1][i].item()) if isinstance(orig_sizes[1], torch.Tensor) else int(orig_sizes[i][1])

                pred_mask = (pred[i, 0] * 255).astype(np.uint8)

                # Resize back to original image size
                pred_mask_resized = cv2.resize(pred_mask, (SAVE_MASK_SIZE, SAVE_MASK_SIZE), interpolation=cv2.INTER_NEAREST)

                # Save ONLY predicted mask with SAME filename
                save_path = os.path.join(save_dir, img_name)
                cv2.imwrite(save_path, pred_mask_resized)

                if has_gt:
                    single_out = outputs[i:i+1]
                    single_gt = masks[i:i+1]

                    dice = dice_coefficient(single_out, single_gt)
                    iou = iou_score(single_out, single_gt)

                    image_records.append({
                        "filename": img_name,
                        "image_path": img_paths[i],
                        "mask_path": mask_paths[i],
                        "Dice (DSC)": dice,
                        "IoU (Jaccard)": iou
                    })

    # Save image-wise metrics
    if has_gt and len(image_records) > 0:
        df = pd.DataFrame(image_records)
        df.to_csv(os.path.join(LOG_DIR, f"{split_name}_imagewise_metrics.csv"), index=False)

        avg_record = {
            "Split": split_name,
            "Average Dice (DSC)": df["Dice (DSC)"].mean(),
            "Average IoU (Jaccard)": df["IoU (Jaccard)"].mean(),
            "Num Images": len(df)
        }

        avg_df = pd.DataFrame([avg_record])
        avg_df.to_csv(os.path.join(LOG_DIR, f"{split_name}_average_metrics.csv"), index=False)

# ======================================================
# TRAIN / VALIDATE
# ======================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    epoch_dice = []
    epoch_iou = []

    for imgs, masks, _, _, _, _ in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_dice.append(dice_coefficient(outputs, masks))
        epoch_iou.append(iou_score(outputs, masks))

    return (
        epoch_loss / len(loader),
        np.mean(epoch_dice),
        np.mean(epoch_iou)
    )


def validate(model, loader, criterion):
    model.eval()
    losses, dices, ious = [], [], []

    with torch.no_grad():
        for imgs, masks, _, _, _, _ in tqdm(loader, desc="Validation"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)

            loss = criterion(outputs, masks)
            losses.append(loss.item())
            dices.append(dice_coefficient(outputs, masks))
            ious.append(iou_score(outputs, masks))

    return np.mean(losses), np.mean(dices), np.mean(ious)

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
    # SAFE TRAIN/VAL SPLIT
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    jsrt_loader = DataLoader(jsrt_dataset, batch_size=1, shuffle=False, num_workers=0)
    synth_loader = DataLoader(synth_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("\n==================== MODEL ====================")
    model = UNet2015(in_channels=1, out_channels=1, dropout_p=DROPOUT_P).to(DEVICE)

    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)

    # ORIGINAL U-NET-STYLE TRAINING
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    best_val_dice = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    patience_counter = 0
    epoch_records = []

    print("\n==================== TRAINING ====================")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion)

        scheduler.step(val_dice)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Train IoU: {train_iou:.4f} || "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        epoch_records.append({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Dice (DSC)": train_dice,
            "Train IoU (Jaccard)": train_iou,
            "Val Loss": val_loss,
            "Val Dice (DSC)": val_dice,
            "Val IoU (Jaccard)": val_iou,
            "Learning Rate": optimizer.param_groups[0]["lr"]
        })

        # Save epoch-wise metrics LIVE after every epoch
        epoch_df = pd.DataFrame(epoch_records)
        epoch_df.to_csv(os.path.join(LOG_DIR, "epoch_metrics.csv"), index=False)
        
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

        # Best model + early stopping
        if val_dice > best_val_dice + MIN_DELTA:
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
            print(f"No significant Dice improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("\nEarly stopping triggered based on Dice.")
            break

    # Save final weights
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "final_model_weights.pth"))

    # Load best model for final evaluation
    model.load_state_dict(best_model_wts)

    # Save epoch-wise metrics
    epoch_df = pd.DataFrame(epoch_records)
    epoch_df.to_csv(os.path.join(LOG_DIR, "epoch_metrics.csv"), index=False)

    print("\n==================== SAVING OUTPUTS ====================")

    evaluate_and_save_predictions(model, train_loader, TRAIN_PRED_DIR, "train", has_gt=True)
    evaluate_and_save_predictions(model, val_loader, VAL_PRED_DIR, "val", has_gt=True)
    evaluate_and_save_predictions(model, test_loader, TEST_PRED_DIR, "test", has_gt=True)
    evaluate_and_save_predictions(model, jsrt_loader, JSRT_PRED_DIR, "jsrt", has_gt=True)
    evaluate_and_save_predictions(model, synth_loader, SYNTH_PRED_DIR, "synth", has_gt=False)

    print("\n==================== DONE ====================")
    print(f"Best model weights saved to: {os.path.join(WEIGHTS_DIR, 'best_model_weights.pth')}")
    print(f"Final model weights saved to: {os.path.join(WEIGHTS_DIR, 'final_model_weights.pth')}")
    print(f"Predictions saved in: {PRED_DIR}")

if __name__ == "__main__":
    main()