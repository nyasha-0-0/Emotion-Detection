import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from joblib import dump
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURATION =A=========
DATA_DIR = "dataset"          # folder containing anger/, happy/, sad/, surprise/
IMG_SIZE = (64, 64)           # resize all images to this size (eye region)
EMOTIONS = ['anger', 'happy', 'sad', 'surprise']
RANDOM_STATE = 42

# ---------- Feature extraction parameters ----------
# HOG
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# Multi‑scale LBP (we use 3 scales)
LBP_RADIUS = [1, 3, 5]
LBP_N_POINTS = [8, 16, 24]

# Gabor filter bank
GABOR_KERNEL_SIZES = [31, 31]    # large enough for eye region
GABOR_THETAS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GABOR_FREQS = [0.1, 0.3, 0.5]

# Spatial grids (divide image into 2x2 cells → 4 grids)
SPATIAL_GRID_CELLS = (2, 2)

# Edge orientation histograms (for Canny edges)
EDGE_HIST_BINS = 8

# ========== FEATURE EXTRACTION FUNCTIONS ==========
def hog_features(img):
    """Extract HOG features from a grayscale image."""
    features = hog(img, orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   visualize=False, feature_vector=True)
    return features

def multi_scale_lbp_features(img):
    """Multi‑scale Local Binary Patterns – concatenated histograms."""
    features = []
    for radius, n_points in zip(LBP_RADIUS, LBP_N_POINTS):
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        # Histogram with 256 bins (for uniform LBP)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)
        features.extend(hist)
    return np.array(features)

def gabor_bank_features(img):
    """Apply a bank of Gabor filters and return mean + std of each response."""
    features = []
    for theta in GABOR_THETAS:
        for freq in GABOR_FREQS:
            # Real part only (or both real+imag – here we use real for simplicity)
            real, _ = gabor(img, frequency=freq, theta=theta,
                            sigma_x=3, sigma_y=3, n_stds=3)
            features.append(real.mean())
            features.append(real.std())
    return np.array(features)

def spatial_grid_features(img):
    """
    Divide image into a grid; for each cell compute:
    - mean intensity
    - standard deviation
    - edge density (from Canny)
    """
    h, w = img.shape
    cell_h = h // SPATIAL_GRID_CELLS[0]
    cell_w = w // SPATIAL_GRID_CELLS[1]
    features = []
    edges = cv2.Canny(img, 50, 150)   # edge map
    for i in range(SPATIAL_GRID_CELLS[0]):
        for j in range(SPATIAL_GRID_CELLS[1]):
            cell = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            edge_cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            features.append(cell.mean())
            features.append(cell.std())
            features.append(np.sum(edge_cell > 0) / cell.size)  # edge density
    return np.array(features)

def edge_orientation_histogram(img):
    """Compute histogram of edge directions from Sobel gradients."""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    orientation = np.arctan2(sobely, sobelx) * (180 / np.pi) % 180
    # Only consider pixels with sufficient gradient magnitude
    mask = magnitude > magnitude.mean()
    hist, _ = np.histogram(orientation[mask], bins=EDGE_HIST_BINS, range=(0, 180), density=True)
    return hist

def extract_all_features(img_gray):
    """Combine all features into one feature vector."""
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    hog_feat = hog_features(img_resized)
    lbp_feat = multi_scale_lbp_features(img_resized)
    gabor_feat = gabor_bank_features(img_resized)
    spatial_feat = spatial_grid_features(img_resized)
    edge_hist_feat = edge_orientation_histogram(img_resized)
    
    return np.concatenate([hog_feat, lbp_feat, gabor_feat, spatial_feat, edge_hist_feat])

# ========== LOAD DATASET ==========
def load_data(data_dir):
    X, y = [], []
    for label_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_dir):
            print(f"Warning: {emotion_dir} not found. Skipping.")
            continue
        img_files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        for img_file in tqdm(img_files, desc=f"Loading {emotion}"):
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            features = extract_all_features(img)
            X.append(features)
            y.append(label_idx)
    return np.array(X), np.array(y)

print("Loading and extracting features... (this may take several minutes)")
X, y = load_data(DATA_DIR)
print(f"Dataset shape: {X.shape}, labels: {len(np.unique(y))}")

# ========== TRAIN & EVALUATE ==========
# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Standardise features (important for distance‑based classifiers but also helps Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest (a robust pure‑ML classifier)
clf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=EMOTIONS))

# Cross‑validation (5‑fold)
cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\n5‑fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix – Eye Region Emotions")
plt.show()

# Save model and scaler
dump(clf, "eye_emotion_model.pkl")
dump(scaler, "eye_emotion_scaler.pkl")
print("\nModel saved as 'eye_emotion_model.pkl' and scaler as 'eye_emotion_scaler.pkl'")
