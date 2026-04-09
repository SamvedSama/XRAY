import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchvision import models

# ======================================================
# CONFIG
# ======================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

WEIGHTS_PATH = "outputs/weights/best_model_weights.pth"

INPUT_FOLDER = "inference_images"
OUTPUT_FOLDER = "inference_outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================================================
# MODEL BLOCKS (SAME AS TRAIN)
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
        return self.relu(out)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
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
# MODEL
# ======================================================

class DDRUNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.densenet201(weights=None)
        features = backbone.features

        self.stem = nn.Sequential(
            features.conv0,
            features.norm0,
            features.relu0
        )

        self.pool0 = features.pool0

        self.denseblock1 = features.denseblock1
        self.transition1 = features.transition1

        self.denseblock2 = features.denseblock2
        self.transition2 = features.transition2

        self.denseblock3 = features.denseblock3
        self.transition3 = features.transition3

        self.denseblock4 = features.denseblock4
        self.norm5 = features.norm5

        self.bottleneck = nn.Sequential(
            ConvBNReLU(1920, 1024, 3, dilation=2),
            ConvBNReLU(1024, 1024, 3, dilation=2)
        )

        # FIXED CHANNELS (IMPORTANT)
        self.dec1 = DecoderBlock(1024, 1792, 512)
        self.dec2 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec4 = DecoderBlock(128, 64, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        s1 = self.stem(x)
        x = self.pool0(s1)

        x = self.denseblock1(x)
        s2 = x
        x = self.transition1(x)

        x = self.denseblock2(x)
        s3 = x
        x = self.transition2(x)

        x = self.denseblock3(x)
        s4 = x
        x = self.transition3(x)

        x = self.denseblock4(x)
        x = self.norm5(x)

        b = self.bottleneck(x)

        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)

        out = self.final_up(d4)
        out = self.final_conv(out)

        return out

# ======================================================
# INFERENCE
# ======================================================

def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    tensor = torch.tensor(img_rgb / 255.0).permute(2, 0, 1).float()
    return tensor.unsqueeze(0), img


def main():
    model = DDRUNet().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    for file in tqdm(os.listdir(INPUT_FOLDER)):
        path = os.path.join(INPUT_FOLDER, file)

        img_tensor, original = preprocess(path)
        img_tensor = img_tensor.to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.sigmoid(output)
            pred = (pred > 0.5).float().cpu().numpy()[0, 0]

        pred_mask = (pred * 255).astype(np.uint8)

        combined = np.hstack([original, pred_mask])
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, file), combined)


if __name__ == "__main__":
    main()