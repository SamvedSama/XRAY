# 1. Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install everything else
pip install -r requirements.txt

# 3. Train
python train_lung_seg.py

# 4. Infer
python infer.py --image chest_xray.png --weights outputs/weights/best_model_weights.pth


The infer.py script produces three files per input: the binary mask, a probability heatmap, and a green-overlay blend on the original X-ray.
You can also call infer_single() directly from any other Python script — it's importable from train_lung_seg.py:

pythonfrom train_lung_seg import infer_single
mask, prob = infer_single("xray.png", "outputs/weights/best_model_weights.pth")