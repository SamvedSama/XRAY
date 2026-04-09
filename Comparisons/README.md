# Real vs Synthetic Chest X-Ray Comparison

A complete quantitative analysis pipeline comparing real and synthetic chest X-ray images
across intensity, quality, texture, frequency, and deep feature dimensions.

## Quick Start

```bash
pip install -r requirements.txt

# Place your images in:
#   real/       ← real X-ray PNGs / JPGs
#   synthetic/  ← synthetic X-ray PNGs / JPGs

python xray_comparison.py
```

All plots are saved to `results/`.

## What Gets Generated

| File | Analysis |
|------|----------|
| `00_summary_dashboard.png` | 6-metric bar chart overview |
| `01_intensity_metrics.png` | Histogram · Boxplots · Scatter (mean vs std) |
| `02_quality_metrics.png`   | MSE · PSNR · SSIM distributions |
| `03a_glcm_heatmap.png`     | GLCM co-occurrence heatmaps |
| `03b_texture_analysis.png` | Haralick features · Edge density |
| `04_frequency_analysis.png`| FFT spectra · Radial power spectrum |
| `05_feature_embeddings.png`| HOG → PCA & t-SNE embeddings |

## Supported Image Formats
`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

## Notes
- Images are auto-converted to grayscale and resized to 256×256.
- Quality metrics (MSE/PSNR/SSIM) use approximate index-based pairings for unpaired datasets.
- HOG descriptors are used as CNN-surrogate features (no GPU required).
