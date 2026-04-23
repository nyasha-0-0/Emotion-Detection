import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from joblib import dump, load
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========== ENHANCED CONFIGURATION ==========
DATA_DIR = "dataset"
IMG_SIZE = (128, 128)  # Larger for better detail
EMOTIONS = ['anger', 'happy', 'sad', 'surprise']
RANDOM_STATE = 42

def enhance_eye_region(img):
    """Enhance eye region for better feature extraction"""
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    
    # Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def extract_enhanced_features(img_gray):
    """Extract more robust features"""
    # Resize and enhance
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    img_enhanced = enhance_eye_region(img_resized)
    
    # 1. HOG with optimized parameters
    hog_feat = hog(img_enhanced, orientations=12, pixels_per_cell=(6, 6),
                   cells_per_block=(2, 2), visualize=False, feature_vector=True)
    
    # 2. Multi-scale LBP (better for texture)
    lbp_features = []
    for radius, n_points in [(1, 8), (2, 16), (3, 24)]:
        lbp = local_binary_pattern(img_enhanced, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_points + 3), density=True)
        lbp_features.extend(hist)
    
    # 3. Gabor filters (capture eye muscle movements)
    gabor_features = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.1, 0.3, 0.5]:
            gabor_real = cv2.getGaborKernel((21,21), 4.0, theta, freq, 0.5, 0)
            filtered = cv2.filter2D(img_enhanced, cv2.CV_32F, gabor_real)
            gabor_features.extend([filtered.mean(), filtered.std(), filtered.var()])
    
    # 4. Eye-specific features (detect eye shapes)
    # Find potential eye regions (dark areas)
    _, binary = cv2.threshold(img_enhanced, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    eye_features = []
    if contours:
        # Largest dark region (potential eye)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        eye_features.extend([area / (IMG_SIZE[0]*IMG_SIZE[1]), circularity, len(contours)])
    else:
        eye_features.extend([0, 0, 0])
    
    # 5. Edge orientation histograms (eyebrow positions matter)
    edges = cv2.Canny(img_enhanced, 30, 100)
    sobelx = cv2.Sobel(img_enhanced, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_enhanced, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
    edge_hist, _ = np.histogram(orientation[magnitude > magnitude.mean()], 
                                 bins=36, range=(-180, 180), density=True)
    
    # 6. Intensity statistics (important for emotion)
    intensity_stats = [
        img_enhanced.mean(), img_enhanced.std(), 
        np.percentile(img_enhanced, 25), np.percentile(img_enhanced, 75),
        img_enhanced.max() - img_enhanced.min()
    ]
    
    # Combine all features
    features = np.concatenate([
        hog_feat, lbp_features, gabor_features, 
        eye_features, edge_hist, intensity_stats
    ])
    
    return features

# Data augmentation
def augment_image(img):
    """Create augmented versions of images"""
    augmented = []
    
    # Original
    augmented.append(img)
    
    # Horizontal flip (eyes are symmetric)
    augmented.append(cv2.flip(img, 1))
    
    # Small rotations
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1)
    augmented.append(cv2.warpAffine(img, M, (w, h)))
    
    M = cv2.getRotationMatrix2D((w/2, h/2), -5, 1)
    augmented.append(cv2.warpAffine(img, M, (w, h)))
    
    # Brightness adjustment
    augmented.append(cv2.convertScaleAbs(img, alpha=1.2, beta=10))
    augmented.append(cv2.convertScaleAbs(img, alpha=0.8, beta=-10))
    
    return augmented

# Load dataset with augmentation
def load_data_augmented():
    X, y = [], []
    
    for label_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(DATA_DIR, emotion)
        if not os.path.isdir(emotion_dir):
            print(f"Warning: {emotion_dir} not found")
            continue
        
        img_files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Loading {len(img_files)} images from {emotion}...")
        
        for img_file in tqdm(img_files, desc=emotion):
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Augment and extract features
            augmented_imgs = augment_image(img)
            for aug_img in augmented_imgs:
                features = extract_enhanced_features(aug_img)
                X.append(features)
                y.append(label_idx)
    
    return np.array(X), np.array(y)

# Load dataset without augmentation (for comparison)
def load_data_simple():
    X, y = [], []
    for label_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(DATA_DIR, emotion)
        if not os.path.isdir(emotion_dir):
            continue
        
        img_files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        for img_file in tqdm(img_files, desc=emotion):
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            features = extract_enhanced_features(img)
            X.append(features)
            y.append(label_idx)
    
    return np.array(X), np.array(y)

# Train with hyperparameter tuning
def train_optimized_model(X_train, y_train):
    print("\n🔧 Optimizing model with Grid Search...")
    
    # Random Forest with hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

# Live detection with improved processing
def live_detection_improved(model, scaler):
    print("\n🎥 IMPROVED Live Detection")
    print("="*50)
    print("Tips for better accuracy:")
    print("1. Look directly at camera")
    print("2. Ensure good lighting")
    print("3. Keep face centered")
    print("4. Press 'q' to quit, 's' to save")
    print("="*50)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Variables for smoothing predictions
    prediction_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Better ROI for eyes (adjust based on face detection)
        roi_x = w // 4
        roi_y = h // 3
        roi_w = w // 2
        roi_h = h // 5
        
        eye_region = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        if eye_region.size > 0:
            # Extract features
            features = extract_enhanced_features(eye_region)
            features_scaled = scaler.transform([features])
            
            # Get prediction probabilities
            probs = model.predict_proba(features_scaled)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            emotion = EMOTIONS[pred_idx]
            
            # Smooth predictions (average last 5)
            prediction_history.append((pred_idx, confidence))
            if len(prediction_history) > 5:
                prediction_history.pop(0)
            
            # Get smoothed prediction
            from collections import Counter
            smoothed_idx = Counter([p[0] for p in prediction_history]).most_common(1)[0][0]
            smoothed_confidence = np.mean([p[1] for p in prediction_history if p[0] == smoothed_idx])
            smoothed_emotion = EMOTIONS[smoothed_idx]
            
            # Color mapping
            colors = {
                'anger': (0, 0, 255),      # Red
                'happy': (0, 255, 255),     # Yellow
                'sad': (255, 0, 0),         # Blue
                'surprise': (0, 255, 0)     # Green
            }
            color = colors.get(smoothed_emotion, (255, 255, 255))
            
            # Draw ROI
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), color, 3)
            
            # Display emotion with confidence
            cv2.putText(frame, f"EMOTION: {smoothed_emotion.upper()}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Confidence: {smoothed_confidence:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int(smoothed_confidence * 200)
            cv2.rectangle(frame, (10, 110), (10 + bar_width, 125), color, -1)
            cv2.rectangle(frame, (10, 110), (210, 125), (255, 255, 255), 1)
            
            # Show probabilities for all emotions
            y_offset = 160
            for idx, (emotion_name, prob) in enumerate(zip(EMOTIONS, probs)):
                prob_color = colors.get(emotion_name, (200, 200, 200))
                bar_w = int(prob * 150)
                cv2.rectangle(frame, (10, y_offset + idx*25), (10 + bar_w, y_offset + idx*25 + 15), prob_color, -1)
                cv2.putText(frame, f"{emotion_name}: {prob:.2f}", (170, y_offset + idx*25 + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, "Press 'q' quit | 's' save", (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Enhanced Emotion Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'captured_{smoothed_emotion}_{smoothed_confidence:.2f}.png', frame)
            print(f"Saved: {smoothed_emotion} with {smoothed_confidence:.2f} confidence")
    
    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("ENHANCED EMOTION DETECTION SYSTEM")
    print("="*60)
    
    # Ask user for mode
    print("\nSelect mode:")
    print("1. Train model (with data augmentation)")
    print("2. Load existing model and run live detection")
    print("3. Train and test with visualization")
    
    mode = input("\nEnter choice (1-3): ").strip()
    
    if mode == '1' or mode == '3':
        print("\n📂 Loading dataset with augmentation...")
        X, y = load_data_augmented()
        print(f"✅ Loaded {len(X)} samples (augmented)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=RANDOM_STATE,
                                                            stratify=y)
        
        # Scale (important for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Optional: PCA to reduce dimensions
        print("\n📉 Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=0.95)  # Keep 95% variance
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        print(f"Reduced from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]} features")
        
        # Train optimized model
        best_model = train_optimized_model(X_train_pca, y_train)
        
        # Evaluate
        y_pred = best_model.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print(f"✅ FINAL TEST ACCURACY: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print("="*50)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=EMOTIONS))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix - Accuracy: {accuracy:.3f}')
        plt.colorbar()
        tick_marks = np.arange(len(EMOTIONS))
        plt.xticks(tick_marks, EMOTIONS, rotation=45)
        plt.yticks(tick_marks, EMOTIONS)
        
        # Add text annotations
        for i in range(len(EMOTIONS)):
            for j in range(len(EMOTIONS)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('improved_confusion_matrix.png', dpi=150)
        plt.show()
        
        # Save model
        dump(best_model, "improved_emotion_model.pkl")
        dump(scaler, "improved_scaler.pkl")
        dump(pca, "improved_pca.pkl")
        print("\n💾 Model saved as 'improved_emotion_model.pkl'")
        
        if mode == '3':
            # Test on sample images
            print("\n📸 Testing on sample images...")
            for emotion in EMOTIONS:
                test_dir = os.path.join(DATA_DIR, emotion)
                if os.path.exists(test_dir):
                    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
                    if test_files:
                        test_img = test_files[0]
                        img_path = os.path.join(test_dir, test_img)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        features = extract_enhanced_features(img)
                        features_scaled = scaler.transform([features])
                        features_pca = pca.transform(features_scaled)
                        pred = best_model.predict(features_pca)[0]
                        print(f"  {emotion} image → Predicted: {EMOTIONS[pred]}")
    
    elif mode == '2':
        # Load existing model
        try:
            best_model = load("improved_emotion_model.pkl")
            scaler = load("improved_scaler.pkl")
            pca = load("improved_pca.pkl")
            print("✅ Model loaded successfully!")
            live_detection_improved(best_model, scaler)
        except:
            print("❌ Model not found. Please train first (option 1)")
    
    else:
        print("Invalid choice!")
