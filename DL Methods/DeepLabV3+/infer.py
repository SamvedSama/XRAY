"""
Inference script — Lung Segmentation DeepLabV3+
Usage:
    python infer.py --image path/to/xray.png --weights outputs/weights/best_model_weights.pth
    python infer.py --image path/to/xray.png --weights outputs/weights/best_model_weights.pth --save_dir results/
    python infer.py --folder path/to/xrays/  --weights outputs/weights/best_model_weights.pth --save_dir results/

Optional flags:
    --size      256         Input resolution (must match training)
    --threshold 0.5         Binarisation threshold
    --no_clahe              Disable CLAHE (disable if not used in training)
    --device    cuda/cpu    Force a device
"""

import os
import argparse
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────
def build_model():
    return smp.DeepLabV3Plus(
        encoder_name    = "resnet50",
        encoder_weights = None,      # weights loaded from file
        in_channels     = 1,
        classes         = 1,
        activation      = None,
    )


def load_model(weights_path, device):
    model = build_model().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"[model] loaded → {weights_path}")
    return model


# ──────────────────────────────────────────────────────────────
# Single image
# ──────────────────────────────────────────────────────────────
def preprocess(image_path, image_size, use_clahe):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape
    img  = cv2.resize(img, (image_size, image_size))
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img   = clahe.apply(img)
    tensor = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, h, w


def run_inference(model, tensor, device, threshold, orig_h, orig_w):
    tensor = tensor.to(device)
    with torch.no_grad():
        logit    = model(tensor)
        prob_map = torch.sigmoid(logit)[0, 0].cpu().numpy()   # H×W float

    pred_mask = ((prob_map > threshold) * 255).astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    prob_map  = cv2.resize(prob_map,  (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return pred_mask, prob_map


def overlay(original_path, pred_mask):
    """Blend the original X-ray with a green-tinted predicted lung mask."""
    orig = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    green = np.zeros_like(orig_bgr)
    green[:, :, 1] = pred_mask          # green channel
    blended = cv2.addWeighted(orig_bgr, 0.7, green, 0.3, 0)
    return blended


def save_outputs(original_path, pred_mask, prob_map, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(original_path))[0]

    mask_path    = os.path.join(save_dir, f"{stem}_mask.png")
    prob_path    = os.path.join(save_dir, f"{stem}_prob.png")
    overlay_path = os.path.join(save_dir, f"{stem}_overlay.png")

    cv2.imwrite(mask_path,    pred_mask)
    cv2.imwrite(prob_path,    (prob_map * 255).astype(np.uint8))
    cv2.imwrite(overlay_path, overlay(original_path, pred_mask))

    print(f"  mask    → {mask_path}")
    print(f"  prob    → {prob_path}")
    print(f"  overlay → {overlay_path}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Lung segmentation inference")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image",  type=str, help="Path to a single X-ray image")
    g.add_argument("--folder", type=str, help="Path to a folder of X-ray images")

    p.add_argument("--weights",   required=True, help="Path to best_model_weights.pth")
    p.add_argument("--save_dir",  default="infer_outputs", help="Where to save results")
    p.add_argument("--size",      type=int,   default=256,   help="Input size (default 256)")
    p.add_argument("--threshold", type=float, default=0.5,   help="Binarisation threshold")
    p.add_argument("--no_clahe",  action="store_true",       help="Disable CLAHE preprocessing")
    p.add_argument("--device",    default=None,              help="'cuda' or 'cpu'")
    return p.parse_args()


def main():
    args      = parse_args()
    device    = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_clahe = not args.no_clahe

    print(f"[config] device={device}  size={args.size}  "
          f"threshold={args.threshold}  clahe={use_clahe}")

    model = load_model(args.weights, device)

    # Collect images
    if args.image:
        paths = [args.image]
    else:
        paths = [
            os.path.join(args.folder, f)
            for f in sorted(os.listdir(args.folder))
            if os.path.splitext(f)[1].lower() in SUPPORTED
        ]
        print(f"[folder] found {len(paths)} images in {args.folder}")

    for img_path in paths:
        print(f"\n[infer] {img_path}")
        try:
            tensor, h, w = preprocess(img_path, args.size, use_clahe)
            mask, prob   = run_inference(model, tensor, device, args.threshold, h, w)
            save_outputs(img_path, mask, prob, args.save_dir)
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n[done]")


if __name__ == "__main__":
    main()
