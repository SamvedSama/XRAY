# ============================================
# IMPORTS
# ============================================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from skimage.feature import graycomatrix, graycoprops
from sklearn.manifold import TSNE

# ============================================
# PATH CONFIG (FROM YOUR CODE)
# ============================================

ROOT = "Normalized Data"
COVIDQU_PATH = os.path.join(ROOT, "covidqu", "Lung Segmentation Data")
JSRT_PATH = os.path.join(ROOT, "jsrt")
SYNTH_PATH = os.path.join(ROOT, "synth", "cxr")

# ============================================
# DATA COLLECTION (REUSED)
# ============================================

def collect_covidqu(split):
    imgs = []
    split_path = os.path.join(COVIDQU_PATH, split)

    for category in ["Normal", "COVID-19", "Non-COVID"]:
        img_dir = os.path.join(split_path, category, "images")
        for file in os.listdir(img_dir):
            imgs.append(os.path.join(img_dir, file))

    return imgs


def collect_jsrt():
    img_dir = os.path.join(JSRT_PATH, "cxr")
    return [os.path.join(img_dir, f) for f in os.listdir(img_dir)]


def collect_synth():
    return [os.path.join(SYNTH_PATH, f) for f in os.listdir(SYNTH_PATH)]

# ============================================
# PREPROCESS
# ============================================

def load_image(path, size=256):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))
    return img / 255.0

# ============================================
# LOAD DATA
# ============================================

real_paths = collect_covidqu("Train") + collect_jsrt()
synthetic_paths = collect_synth()

real_images = [load_image(p) for p in tqdm(real_paths, desc="Loading REAL")]
synthetic_images = [load_image(p) for p in tqdm(synthetic_paths, desc="Loading SYNTH")]

# ============================================
# CREATE RESULTS DIR
# ============================================

os.makedirs("results1", exist_ok=True)

# ============================================
# GLOBAL METRICS
# ============================================

def global_metrics(img):
    mean = np.mean(img)
    std = np.std(img)
    entropy = -np.sum(img * np.log2(img + 1e-8))
    snr = mean / (std + 1e-8)
    return mean, std, entropy, snr

real_m = [global_metrics(img) for img in real_images]
syn_m = [global_metrics(img) for img in synthetic_images]

real_mean = [m[0] for m in real_m]
syn_mean = [m[0] for m in syn_m]

real_std = [m[1] for m in real_m]
syn_std = [m[1] for m in syn_m]

# Histogram
plt.hist(np.concatenate([img.flatten() for img in real_images]), bins=50, alpha=0.5, label="Real")
plt.hist(np.concatenate([img.flatten() for img in synthetic_images]), bins=50, alpha=0.5, label="Synthetic")
plt.legend()
plt.title("Pixel Intensity Distribution")
plt.savefig("results1/histogram.png")
plt.close()

# Boxplot
sns.boxplot(data=[real_mean, syn_mean])
plt.xticks([0,1], ["Real","Synthetic"])
plt.title("Brightness")
plt.savefig("results1/brightness.png")
plt.close()

# ============================================
# TEXTURE ANALYSIS
# ============================================

def texture_features(img):
    img_uint8 = (img * 255).astype(np.uint8)

    glcm = graycomatrix(img_uint8, [1], [0], 256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]

    return contrast, energy

real_tex = [texture_features(img) for img in real_images]
syn_tex = [texture_features(img) for img in synthetic_images]

real_contrast = [f[0] for f in real_tex]
real_energy = [f[1] for f in real_tex]

syn_contrast = [f[0] for f in syn_tex]
syn_energy = [f[1] for f in syn_tex]

plt.scatter(real_contrast, real_energy, label="Real")
plt.scatter(syn_contrast, syn_energy, label="Synthetic")
plt.legend()
plt.xlabel("Contrast")
plt.ylabel("Energy")
plt.title("GLCM Features")
plt.savefig("results1/glcm.png")
plt.close()

# ============================================
# EDGE DENSITY
# ============================================

def edge_density(img):
    edges = cv2.Canny((img*255).astype(np.uint8), 100, 200)
    return np.sum(edges) / edges.size

real_edges = [edge_density(img) for img in real_images]
syn_edges = [edge_density(img) for img in synthetic_images]

plt.boxplot([real_edges, syn_edges])
plt.xticks([1,2], ["Real","Synthetic"])
plt.title("Edge Density")
plt.savefig("results1/edges.png")
plt.close()

# ============================================
# FREQUENCY ANALYSIS
# ============================================

def fft_energy(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1)

    h, w = mag.shape
    return np.sum(mag[h//4:, w//4:])

real_fft = [fft_energy(img) for img in real_images]
syn_fft = [fft_energy(img) for img in synthetic_images]

plt.boxplot([real_fft, syn_fft])
plt.xticks([1,2], ["Real","Synthetic"])
plt.title("High Frequency Energy")
plt.savefig("results1/frequency.png")
plt.close()

# ============================================
# t-SNE
# ============================================

def extract_features(img):
    return img.flatten()

features = []
labels = []

for img in real_images:
    features.append(extract_features(img))
    labels.append(0)

for img in synthetic_images:
    features.append(extract_features(img))
    labels.append(1)

features = np.array(features)

tsne = TSNE(n_components=2, random_state=42)
embedding = tsne.fit_transform(features)

embedding = np.array(embedding)

real_emb = embedding[np.array(labels)==0]
syn_emb = embedding[np.array(labels)==1]

plt.scatter(real_emb[:,0], real_emb[:,1], label="Real")
plt.scatter(syn_emb[:,0], syn_emb[:,1], label="Synthetic")
plt.legend()
plt.title("t-SNE Feature Space")
plt.savefig("results1/tsne.png")
plt.close()

print("✅ Analysis complete using YOUR dataset!")