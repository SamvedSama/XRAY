import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn

# ======================================================
# CONFIG
# ======================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
DEFAULT_WEIGHTS = "outputs_unet2015/weights/best_model_weights.pth"
DEFAULT_OUTPUT_DIR = "single_inference_outputs"

os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# ======================================================
# MODEL (MUST MATCH TRAINING EXACTLY)
# ======================================================

class DoubleConv(nn.Module):
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
# METRICS
# ======================================================

def dice_score(pred_mask, gt_mask, smooth=1.0):
    pred = pred_mask.astype(np.float32).flatten()
    gt = gt_mask.astype(np.float32).flatten()

    intersection = np.sum(pred * gt)
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    return float(dice)


def iou_score(pred_mask, gt_mask, smooth=1.0):
    pred = pred_mask.astype(np.float32).flatten()
    gt = gt_mask.astype(np.float32).flatten()

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)

# ======================================================
# HELPERS
# ======================================================

def load_model(weights_path):
    model = UNet2015(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_tensor = torch.tensor(img_resized / 255.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    return img, img_tensor, orig_h, orig_w


def preprocess_mask(mask_path, orig_h, orig_w):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    # IMPORTANT:
    # resize GT to original image size first if needed
    if mask.shape[:2] != (orig_h, orig_w):
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    gt_mask_bin = (mask > 127).astype(np.uint8)
    return gt_mask_bin


def predict_mask(model, img_tensor, orig_h, orig_w):
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float().cpu().numpy()[0, 0]

    pred_mask = (pred * 255).astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    pred_mask_bin = (pred_mask > 127).astype(np.uint8)

    return pred_mask, pred_mask_bin


def save_pred_mask(pred_mask, image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, image_name)
    cv2.imwrite(save_path, pred_mask)
    return save_path

# ======================================================
# MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="Single-image inference for trained U-Net")

    parser.add_argument("--mode", type=str, required=True, choices=["real", "synth"],
                        help="real = no GT, synth = GT available")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask", type=str, default=None, help="Path to GT mask (required for synth)")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to trained model weights")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Where to save predicted mask")

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Loading weights from: {args.weights}")

    if args.mode == "synth" and args.mask is None:
        raise ValueError("For synth mode, you must provide --mask path/to/gt_mask.png")

    model = load_model(args.weights)

    # -----------------------------
    # Load and preprocess image
    # -----------------------------
    original_img, img_tensor, orig_h, orig_w = preprocess_image(args.image)

    # -----------------------------
    # Predict
    # -----------------------------
    pred_mask, pred_mask_bin = predict_mask(model, img_tensor, orig_h, orig_w)

    # -----------------------------
    # Save prediction
    # -----------------------------
    save_path = save_pred_mask(pred_mask, args.image, args.output_dir)

    print("\n==================== INFERENCE DONE ====================")
    print(f"Input image      : {args.image}")
    print(f"Saved pred mask  : {save_path}")

    # -----------------------------
    # If synth, compute metrics
    # -----------------------------
    if args.mode == "synth":
        gt_mask_bin = preprocess_mask(args.mask, orig_h, orig_w)

        dice = dice_score(pred_mask_bin, gt_mask_bin)
        iou = iou_score(pred_mask_bin, gt_mask_bin)

        print(f"GT mask          : {args.mask}")
        print(f"Dice (DSC)       : {dice:.6f}")
        print(f"IoU (Jaccard)    : {iou:.6f}")

    else:
        print("Mode             : real")
        print("Dice / IoU       : Not computed (no GT mask provided)")

    print("=======================================================\n")


if __name__ == "__main__":
    main()