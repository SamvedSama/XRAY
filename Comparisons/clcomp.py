"""
=============================================================================
  REAL vs SYNTHETIC CHEST X-RAY COMPARISON  |  Quantitative Analysis Tool
=============================================================================
  Covers:
    1. Data loading & preprocessing
    2. Global intensity metrics  (brightness, contrast, entropy, SNR)
    3. Image quality metrics     (MSE, PSNR, SSIM)
    4. Texture analysis          (GLCM / Haralick + Canny edge density)
    5. Frequency-domain analysis (FFT, radial power spectrum)
    6. Deep feature analysis     (PCA / t-SNE on HOG descriptors)
=============================================================================
"""

# ── Standard / third-party imports ──────────────────────────────────────────
import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from pathlib import Path
from scipy.stats import entropy as scipy_entropy
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse_skimage
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ── Paths (mirrors your experiment script) ───────────────────────────────────
ROOT         = "Normalized Data"
COVIDQU_PATH = os.path.join(ROOT, "covidqu", "Lung Segmentation Data")
JSRT_PATH    = os.path.join(ROOT, "jsrt")
SYNTH_PATH   = os.path.join(ROOT, "synth", "cxr")

REAL_COLOR   = "#E84393"
SYNTH_COLOR  = "#00C2CB"
BASE_DIR     = Path(__file__).parent
RESULTS_DIR  = BASE_DIR / "results"
IMG_SIZE     = (256, 256)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _read_and_preprocess(path: str, size=IMG_SIZE) -> np.ndarray | None:
    """Read one image → grayscale → resize → normalise to [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [WARN] Could not read {path} — skipping.")
        return None
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def collect_covidqu_images(split="Train") -> tuple[list, list]:
    """
    Mirrors collect_covidqu() from the experiment script.
    Returns (images, names) — masks are ignored here (we only need images).
    """
    images, names = [], []
    split_path = os.path.join(COVIDQU_PATH, split)

    for category in ["Normal", "COVID-19", "Non-COVID"]:
        img_dir = os.path.join(split_path, category, "images")
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            fpath = os.path.join(img_dir, fname)
            img = _read_and_preprocess(fpath)
            if img is not None:
                images.append(img)
                names.append(f"covidqu/{split}/{category}/{fname}")

    print(f"  Loaded {len(images)} CovidQU-{split} images")
    return images, names


def collect_jsrt_images() -> tuple[list, list]:
    """Mirrors collect_jsrt() — loads from jsrt/cxr/."""
    img_dir = os.path.join(JSRT_PATH, "cxr")
    images, names = [], []

    for fname in sorted(os.listdir(img_dir)):
        fpath = os.path.join(img_dir, fname)
        img = _read_and_preprocess(fpath)
        if img is not None:
            images.append(img)
            names.append(f"jsrt/{fname}")

    print(f"  Loaded {len(images)} JSRT images")
    return images, names


def collect_synth_images() -> tuple[list, list]:
    """Mirrors collect_synth() — loads from synth/cxr/."""
    images, names = [], []

    for fname in sorted(os.listdir(SYNTH_PATH)):
        fpath = os.path.join(SYNTH_PATH, fname)
        img = _read_and_preprocess(fpath)
        if img is not None:
            images.append(img)
            names.append(f"synth/{fname}")

    print(f"  Loaded {len(images)} synthetic images")
    return images, names


# ═══════════════════════════════════════════════════════════════════════════════
#  2. GLOBAL INTENSITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_intensity_metrics(images: list[np.ndarray]) -> dict:
    """
    Per-image: mean (brightness), std (contrast), Shannon entropy, SNR.
    """
    means, stds, entropies, snrs = [], [], [], []
    for img in images:
        means.append(float(np.mean(img)))
        stds.append(float(np.std(img)))

        # Shannon entropy from 256-bin histogram
        hist, _ = np.histogram(img, bins=256, range=(0, 1))
        hist = hist / hist.sum()                       # normalise to PMF
        entropies.append(float(scipy_entropy(hist + 1e-12, base=2)))

        # SNR = mean / std  (simple radiological definition)
        snr = np.mean(img) / (np.std(img) + 1e-8)
        snrs.append(float(snr))

    return {"mean": means, "std": stds, "entropy": entropies, "snr": snrs}


def plot_intensity(real_m, synth_m, real_imgs, synth_imgs):
    """
    Three panels:
      A) Overlay pixel-intensity histograms
      B) Boxplots of brightness & contrast
      C) Scatter: mean vs std (each point = one image)
    """
    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── A: Histogram overlay ─────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    bins = np.linspace(0, 1, 80)
    real_pixels  = np.concatenate([img.ravel() for img in real_imgs])
    synth_pixels = np.concatenate([img.ravel() for img in synth_imgs])
    ax0.hist(real_pixels,  bins=bins, density=True, alpha=0.55,
             color=REAL_COLOR,  label="Real",      histtype="stepfilled", lw=1.2, edgecolor=REAL_COLOR)
    ax0.hist(synth_pixels, bins=bins, density=True, alpha=0.55,
             color=SYNTH_COLOR, label="Synthetic", histtype="stepfilled", lw=1.2, edgecolor=SYNTH_COLOR)
    ax0.set_xlabel("Pixel Intensity (normalised)")
    ax0.set_ylabel("Density")
    ax0.set_title("A  Pixel Intensity Distribution")
    ax0.legend()

    # ── B: Boxplots ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    data  = [real_m["mean"],  synth_m["mean"],  real_m["std"],  synth_m["std"]]
    ticks = ["Real\nBrightness", "Synth\nBrightness", "Real\nContrast", "Synth\nContrast"]
    colors = [REAL_COLOR, SYNTH_COLOR, REAL_COLOR, SYNTH_COLOR]
    bp = ax1.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color="white", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax1.set_xticks(range(1, 5)); ax1.set_xticklabels(ticks, fontsize=8)
    ax1.set_ylabel("Value")
    ax1.set_title("B  Brightness & Contrast Boxplots")

    # ── C: Scatter mean vs std ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.scatter(real_m["mean"],  real_m["std"],  c=REAL_COLOR,  s=60, alpha=0.8, label="Real",      edgecolors="white", lw=0.5)
    ax2.scatter(synth_m["mean"], synth_m["std"], c=SYNTH_COLOR, s=60, alpha=0.8, label="Synthetic", edgecolors="white", lw=0.5)
    ax2.set_xlabel("Mean Intensity (Brightness)")
    ax2.set_ylabel("Std Dev (Contrast)")
    ax2.set_title("C  Brightness vs Contrast Scatter")
    ax2.legend()

    fig.suptitle("Global Intensity Metrics", fontsize=14, fontweight="bold", y=1.01)
    _save(fig, "01_intensity_metrics.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  3. IMAGE QUALITY METRICS  (MSE · PSNR · SSIM)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_quality_metrics(real_imgs, synth_imgs):
    """
    Pair each real image with the closest synthetic image (by index mod n).
    Computes MSE, PSNR, and SSIM for each pair.

    NOTE: For unpaired datasets the pairing is approximate.
    """
    n = min(len(real_imgs), len(synth_imgs))
    mse_vals, psnr_vals, ssim_vals = [], [], []
    for i in range(n):
        r = real_imgs[i]
        s = synth_imgs[i % len(synth_imgs)]

        m  = float(np.mean((r - s) ** 2))
        p  = float(10 * np.log10(1.0 / (m + 1e-12)))
        sv = float(ssim(r, s, data_range=1.0))

        mse_vals.append(m)
        psnr_vals.append(p)
        ssim_vals.append(sv)

    return {"mse": mse_vals, "psnr": psnr_vals, "ssim": ssim_vals}


def plot_quality(qm):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, key, label, color in zip(
        axes,
        ["mse",    "psnr",   "ssim"],
        ["MSE",    "PSNR (dB)", "SSIM"],
        ["#9B59B6","#F39C12","#27AE60"],
    ):
        vals = qm[key]
        ax.hist(vals, bins=max(5, len(vals)//2 + 1), color=color, alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(vals), color="black", ls="--", lw=1.5, label=f"Mean={np.mean(vals):.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}")
        ax.legend()

    fig.suptitle("Image Quality Metrics (Approximate Pairings)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_quality_metrics.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  4. TEXTURE ANALYSIS  (GLCM / Haralick  +  Edge density)
# ═══════════════════════════════════════════════════════════════════════════════

def _glcm_features(img_float: np.ndarray) -> dict:
    """Extract Haralick texture features from a single image via GLCM."""
    img8 = (img_float * 255).astype(np.uint8)
    # Quantise to 64 levels for tractable GLCM
    img_q = (img8 // 4).astype(np.uint8)
    glcm  = graycomatrix(img_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                          levels=64, symmetric=True, normed=True)
    return {
        "contrast":    float(graycoprops(glcm, "contrast").mean()),
        "energy":      float(graycoprops(glcm, "energy").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "correlation": float(graycoprops(glcm, "correlation").mean()),
    }


def _edge_density(img_float: np.ndarray) -> float:
    """Fraction of pixels detected as edges by Canny."""
    img8  = (img_float * 255).astype(np.uint8)
    edges = cv2.Canny(img8, threshold1=30, threshold2=100)
    return float(np.sum(edges > 0) / edges.size)


def compute_texture_metrics(images: list[np.ndarray]) -> dict:
    feats = {"contrast": [], "energy": [], "homogeneity": [],
             "correlation": [], "edge_density": []}
    for img in images:
        hf = _glcm_features(img)
        for k in ["contrast","energy","homogeneity","correlation"]:
            feats[k].append(hf[k])
        feats["edge_density"].append(_edge_density(img))
    return feats


def plot_glcm_heatmap(real_imgs, synth_imgs):
    """Show normalised GLCM of one representative image from each group."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, img, title, cmap in zip(
        axes,
        [real_imgs[0], synth_imgs[0]],
        ["Real – GLCM (0°)", "Synthetic – GLCM (0°)"],
        ["magma", "cividis"],
    ):
        img8  = ((img * 255) // 4).astype(np.uint8)
        glcm  = graycomatrix(img8, distances=[1], angles=[0],
                              levels=64, symmetric=True, normed=True)
        mat   = glcm[:, :, 0, 0]
        im    = ax.imshow(mat, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("Gray Level j"); ax.set_ylabel("Gray Level i")

    fig.suptitle("GLCM Heatmaps – Texture Co-occurrence Patterns",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03a_glcm_heatmap.png")
    return fig


def plot_texture(real_t, synth_t):
    """Four subplots: Contrast vs Energy scatter, and boxplots for each feature."""
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Scatter: Contrast vs Energy ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(real_t["contrast"],  real_t["energy"],
                c=REAL_COLOR,  s=65, alpha=0.85, label="Real",      edgecolors="white", lw=0.4)
    ax0.scatter(synth_t["contrast"], synth_t["energy"],
                c=SYNTH_COLOR, s=65, alpha=0.85, label="Synthetic", edgecolors="white", lw=0.4)
    ax0.set_xlabel("GLCM Contrast"); ax0.set_ylabel("GLCM Energy")
    ax0.set_title("Contrast vs Energy (GLCM)")
    ax0.legend()

    # ── Boxplots for 4 Haralick features ─────────────────────────────────────
    features = ["contrast", "energy", "homogeneity", "edge_density"]
    titles   = ["GLCM Contrast", "GLCM Energy", "GLCM Homogeneity", "Canny Edge Density"]
    positions = [(0,1),(0,2),(1,0),(1,1)]

    for feat, title, pos in zip(features, titles, positions):
        ax = fig.add_subplot(gs[pos])
        data   = [real_t[feat], synth_t[feat]]
        labels = ["Real", "Synthetic"]
        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="white", lw=2))
        for patch, c in zip(bp["boxes"], [REAL_COLOR, SYNTH_COLOR]):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.set_xticks([1,2]); ax.set_xticklabels(labels)
        ax.set_title(title)

    # ── Edge density scatter ──────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.scatter(real_t["contrast"],  real_t["edge_density"],
                 c=REAL_COLOR,  s=65, alpha=0.85, label="Real",      edgecolors="white", lw=0.4)
    ax_e.scatter(synth_t["contrast"], synth_t["edge_density"],
                 c=SYNTH_COLOR, s=65, alpha=0.85, label="Synthetic", edgecolors="white", lw=0.4)
    ax_e.set_xlabel("GLCM Contrast"); ax_e.set_ylabel("Edge Density")
    ax_e.set_title("Contrast vs Edge Density")
    ax_e.legend()

    fig.suptitle("Texture Analysis – GLCM Haralick Features & Edge Density",
                 fontsize=14, fontweight="bold")
    _save(fig, "03b_texture_analysis.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  5. FREQUENCY-DOMAIN ANALYSIS  (FFT)
# ═══════════════════════════════════════════════════════════════════════════════

def _fft_spectrum(img: np.ndarray) -> np.ndarray:
    """Return log-scaled, centred FFT magnitude spectrum."""
    f    = np.fft.fft2(img)
    fsh  = np.fft.fftshift(f)
    mag  = np.abs(fsh)
    return np.log1p(mag)               # log for dynamic range compression


def _freq_energies(img: np.ndarray) -> tuple[float, float]:
    """
    Split the FFT into low-freq (inner 25%) and high-freq (outer 75%) rings.
    Returns (low_energy, high_energy) as fractions of total power.
    """
    H, W   = img.shape
    f      = np.fft.fftshift(np.fft.fft2(img))
    power  = np.abs(f) ** 2
    total  = power.sum() + 1e-12

    cy, cx = H // 2, W // 2
    Y, X   = np.ogrid[:H, :W]
    dist   = np.sqrt((Y - cy)**2 + (X - cx)**2)
    radius = min(H, W) / 2

    low_mask  = dist <= radius * 0.25
    high_mask = dist >  radius * 0.25

    return float(power[low_mask].sum() / total), float(power[high_mask].sum() / total)


def _radial_power(img: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """1-D radially averaged power spectrum (azimuthal average)."""
    H, W  = img.shape
    f     = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(f) ** 2
    cy, cx = H//2, W//2
    Y, X   = np.ogrid[:H, :W]
    dist   = np.sqrt((Y-cy)**2 + (X-cx)**2).astype(int)
    max_r  = min(H, W)//2
    profile = np.zeros(max_r)
    counts  = np.zeros(max_r)
    for r in range(max_r):
        mask = dist == r
        if mask.any():
            profile[r] = power[mask].mean()
            counts[r]  = mask.sum()
    return profile


def compute_freq_metrics(images: list[np.ndarray]) -> dict:
    lows, highs = [], []
    for img in images:
        lo, hi = _freq_energies(img)
        lows.append(lo); highs.append(hi)
    return {"low_energy": lows, "high_energy": highs}


def plot_frequency(real_imgs, synth_imgs, real_fm, synth_fm):
    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── FFT visual: real sample ───────────────────────────────────────────────
    for col, imgs, title in [(0, real_imgs, "Real – FFT Spectrum"),
                              (1, synth_imgs, "Synthetic – FFT Spectrum")]:
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(_fft_spectrum(imgs[0]), cmap="inferno", aspect="auto")
        ax.set_title(title); ax.axis("off")

    # ── Low vs High energy boxplot ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    bp  = ax2.boxplot(
        [real_fm["low_energy"],  synth_fm["low_energy"],
         real_fm["high_energy"], synth_fm["high_energy"]],
        patch_artist=True, widths=0.45,
        medianprops=dict(color="white", lw=2)
    )
    for patch, c in zip(bp["boxes"],
                        [REAL_COLOR, SYNTH_COLOR, REAL_COLOR, SYNTH_COLOR]):
        patch.set_facecolor(c); patch.set_alpha(0.72)
    ax2.set_xticks([1,2,3,4])
    ax2.set_xticklabels(["Low\nReal","Low\nSynth","High\nReal","High\nSynth"], fontsize=8)
    ax2.set_title("Frequency Energy Distribution")
    ax2.set_ylabel("Fractional Power")

    # ── Radial power spectrum ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    real_rps  = np.array([_radial_power(img) for img in real_imgs])
    synth_rps = np.array([_radial_power(img) for img in synth_imgs])
    freqs     = np.arange(real_rps.shape[1])
    ax3.semilogy(freqs, real_rps.mean(axis=0),  color=REAL_COLOR,  lw=2, label="Real")
    ax3.semilogy(freqs, synth_rps.mean(axis=0), color=SYNTH_COLOR, lw=2, label="Synthetic")
    ax3.fill_between(freqs,
                     real_rps.mean(0) - real_rps.std(0),
                     real_rps.mean(0) + real_rps.std(0),
                     alpha=0.2, color=REAL_COLOR)
    ax3.fill_between(freqs,
                     synth_rps.mean(0) - synth_rps.std(0),
                     synth_rps.mean(0) + synth_rps.std(0),
                     alpha=0.2, color=SYNTH_COLOR)
    ax3.set_xlabel("Spatial Frequency (cycles/image)")
    ax3.set_ylabel("Mean Power (log scale)")
    ax3.set_title("Radial Power Spectrum")
    ax3.legend()

    # ── Original images (bottom row) ─────────────────────────────────────────
    for col, imgs, label in [(0, real_imgs, "Real Sample"),
                              (1, synth_imgs, "Synthetic Sample")]:
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(imgs[0], cmap="gray", aspect="auto")
        ax.set_title(label); ax.axis("off")

    # ── Canny edge maps ───────────────────────────────────────────────────────
    for col, imgs, label in [(2, real_imgs, "Real – Edges"),
                              (3, synth_imgs, "Synthetic – Edges")]:
        ax = fig.add_subplot(gs[1, col])
        img8 = (imgs[0] * 255).astype(np.uint8)
        edges = cv2.Canny(img8, 30, 100)
        ax.imshow(edges, cmap="gray", aspect="auto")
        ax.set_title(label); ax.axis("off")

    fig.suptitle("Frequency Domain Analysis – FFT & Radial Power Spectrum",
                 fontsize=14, fontweight="bold")
    _save(fig, "04_frequency_analysis.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  6. DEEP FEATURE ANALYSIS  (HOG → PCA / t-SNE)
# ═══════════════════════════════════════════════════════════════════════════════

def _hog_feature(img: np.ndarray, cell: int = 16) -> np.ndarray:
    """
    Compute a histogram-of-oriented-gradients descriptor.
    Used as a CNN-surrogate when PyTorch is unavailable.
    """
    img8  = (img * 255).astype(np.uint8)
    h, w  = img8.shape
    gx    = cv2.Sobel(img8, cv2.CV_64F, 1, 0, ksize=3)
    gy    = cv2.Sobel(img8, cv2.CV_64F, 0, 1, ksize=3)
    mag   = np.sqrt(gx**2 + gy**2)
    ang   = np.rad2deg(np.arctan2(gy, gx)) % 180

    n_bins   = 9
    features = []
    for row in range(0, h - cell + 1, cell):
        for col in range(0, w - cell + 1, cell):
            m_cell = mag[row:row+cell, col:col+cell]
            a_cell = ang[row:row+cell, col:col+cell]
            hist, _ = np.histogram(a_cell, bins=n_bins, range=(0, 180),
                                   weights=m_cell)
            features.extend(hist.tolist())
    return np.array(features, dtype=np.float32)


def compute_deep_features(real_imgs, synth_imgs) -> tuple[np.ndarray, np.ndarray]:
    print("  Computing HOG descriptors …")
    real_feats  = np.vstack([_hog_feature(img) for img in real_imgs])
    synth_feats = np.vstack([_hog_feature(img) for img in synth_imgs])
    return real_feats, synth_feats


def plot_embeddings(real_feats, synth_feats):
    """
    PCA (top panel) and t-SNE (bottom panel) of combined feature matrix.
    Each point is one image, coloured by real / synthetic label.
    """
    X      = np.vstack([real_feats,  synth_feats])
    labels = np.array([0]*len(real_feats) + [1]*len(synth_feats))   # 0=real, 1=synth
    colours = np.where(labels == 0, REAL_COLOR, SYNTH_COLOR)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca     = PCA(n_components=2, random_state=0)
    X_pca   = pca.fit_transform(X_sc)
    var_exp = pca.explained_variance_ratio_ * 100

    # ── t-SNE ────────────────────────────────────────────────────────────────
    n_total  = len(X)
    perplexity = min(5, max(2, n_total // 3))   # safe for small datasets
    tsne     = TSNE(n_components=2, random_state=0, perplexity=perplexity,
                    max_iter=1000, init="pca")
    X_tsne   = tsne.fit_transform(X_sc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, emb, title_base, xlabel, ylabel in [
        (axes[0], X_pca,
         "PCA Projection",
         f"PC1 ({var_exp[0]:.1f}% var)",
         f"PC2 ({var_exp[1]:.1f}% var)"),
        (axes[1], X_tsne,
         "t-SNE Projection",
         "t-SNE Dim 1",
         "t-SNE Dim 2"),
    ]:
        for group_label, color, marker, name in [
            (0, REAL_COLOR,  "o", "Real"),
            (1, SYNTH_COLOR, "^", "Synthetic"),
        ]:
            mask = labels == group_label
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=color, s=80, alpha=0.9, marker=marker,
                       label=name, edgecolors="white", lw=0.5)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title_base)
        ax.legend()

    fig.suptitle("Deep Feature Embeddings – HOG Descriptors (PCA & t-SNE)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "05_feature_embeddings.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  7. COMPREHENSIVE SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(real_m, synth_m, real_t, synth_t, real_fm, synth_fm):
    """
    A single 2×3 overview figure showing six key metric comparisons.
    Designed for one-page inclusion in a research report.
    """
    metrics = [
        ("Brightness (Mean)",   real_m["mean"],      synth_m["mean"]),
        ("Contrast (Std Dev)",  real_m["std"],        synth_m["std"]),
        ("Entropy (bits)",      real_m["entropy"],    synth_m["entropy"]),
        ("GLCM Energy",         real_t["energy"],     synth_t["energy"]),
        ("Edge Density",        real_t["edge_density"],synth_t["edge_density"]),
        ("High-Freq Energy",    real_fm["high_energy"],synth_fm["high_energy"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for ax, (title, r_vals, s_vals) in zip(axes, metrics):
        ax.bar(["Real", "Synthetic"],
               [np.mean(r_vals), np.mean(s_vals)],
               color=[REAL_COLOR, SYNTH_COLOR],
               alpha=0.85,
               yerr=[np.std(r_vals), np.std(s_vals)],
               capsize=6,
               error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="black"),
               edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Value")

    fig.suptitle("Summary Dashboard – Real vs Synthetic X-Rays",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "00_summary_dashboard.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, filename: str):
    path = RESULTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")


def print_stats(name: str, vals: list[float]):
    arr = np.array(vals)
    print(f"    {name:25s}  mean={arr.mean():.4f}  std={arr.std():.4f}"
          f"  min={arr.min():.4f}  max={arr.max():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  CHEST X-RAY COMPARISON  |  Real vs Synthetic")
    print("=" * 65)

    # ── 1. Load images ────────────────────────────────────────────────────────
    # ── 1. Load images ────────────────────────────────────────────────────────
    print("\n[1/6]  Loading images …")

    # Real = CovidQU Train + JSRT combined
    covidqu_imgs, covidqu_names = collect_covidqu_images("Train")
    jsrt_imgs,    jsrt_names    = collect_jsrt_images()
    real_imgs  = covidqu_imgs + jsrt_imgs
    real_names = covidqu_names + jsrt_names

    # Synthetic = synth/cxr/
    synth_imgs, synth_names = collect_synth_images()

    print(f"\n  Total REAL images    : {len(real_imgs)}")
    print(f"  Total SYNTHETIC images: {len(synth_imgs)}")

    if len(real_imgs) == 0 or len(synth_imgs) == 0:
        raise RuntimeError("One or both image groups are empty — check your paths.")

    # ── 2. Global intensity metrics ───────────────────────────────────────────
    print("\n[2/6]  Computing intensity metrics …")
    real_m  = compute_intensity_metrics(real_imgs)
    synth_m = compute_intensity_metrics(synth_imgs)

    print("  Real images :")
    for k in real_m:  print_stats(k, real_m[k])
    print("  Synthetic images :")
    for k in synth_m: print_stats(k, synth_m[k])

    plot_intensity(real_m, synth_m, real_imgs, synth_imgs)

    # ── 3. Image quality metrics ──────────────────────────────────────────────
    print("\n[3/6]  Computing quality metrics (approximate pairings) …")
    qm = compute_quality_metrics(real_imgs, synth_imgs)
    print(f"  Mean MSE ={np.mean(qm['mse']):.4f}   "
          f"PSNR={np.mean(qm['psnr']):.2f} dB   "
          f"SSIM={np.mean(qm['ssim']):.4f}")
    plot_quality(qm)

    # ── 4. Texture analysis ───────────────────────────────────────────────────
    print("\n[4/6]  Computing texture (GLCM / Haralick / Canny) …")
    real_t  = compute_texture_metrics(real_imgs)
    synth_t = compute_texture_metrics(synth_imgs)

    print("  Real  – GLCM contrast  : ", np.mean(real_t["contrast"]))
    print("  Synth – GLCM contrast  : ", np.mean(synth_t["contrast"]))
    print("  Real  – edge density   : ", np.mean(real_t["edge_density"]))
    print("  Synth – edge density   : ", np.mean(synth_t["edge_density"]))

    plot_glcm_heatmap(real_imgs, synth_imgs)
    plot_texture(real_t, synth_t)

    # ── 5. Frequency analysis ─────────────────────────────────────────────────
    print("\n[5/6]  Computing frequency domain metrics …")
    real_fm  = compute_freq_metrics(real_imgs)
    synth_fm = compute_freq_metrics(synth_imgs)

    print(f"  Real  – High-Freq Energy : {np.mean(real_fm['high_energy']):.4f}")
    print(f"  Synth – High-Freq Energy : {np.mean(synth_fm['high_energy']):.4f}")

    plot_frequency(real_imgs, synth_imgs, real_fm, synth_fm)

    # ── 6. Deep feature embeddings ────────────────────────────────────────────
    print("\n[6/6]  Computing HOG features & embeddings …")
    real_feats, synth_feats = compute_deep_features(real_imgs, synth_imgs)
    plot_embeddings(real_feats, synth_feats)

    # ── Summary dashboard ─────────────────────────────────────────────────────
    print("\n[+]  Generating summary dashboard …")
    plot_summary_dashboard(real_m, synth_m, real_t, synth_t, real_fm, synth_fm)

    print("\n" + "=" * 65)
    print(f"  All plots saved to '{RESULTS_DIR}/'")
    print("=" * 65)


if __name__ == "__main__":
    main()
