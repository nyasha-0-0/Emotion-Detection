import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump, load
from tkinter import filedialog, Tk
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
DATA_DIR = "dataset"
IMG_SIZE = (96, 96)  # Good balance
EMOTIONS = ['anger', 'happy', 'sad', 'surprise']
RANDOM_STATE = 42

def preprocess_eye_region(img):
    """Advanced preprocessing for eye regions"""
    # Resize
    img = cv2.resize(img, IMG_SIZE)
    
    # Denoise
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Normalize
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    return img

def extract_robust_features(img):
    """Extract features that actually work for eyes"""
    img_proc = preprocess_eye_region(img)
    
    # 1. HOG (good for shape)
    hog_feat = hog(img_proc, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    
    # 2. LBP (good for texture)
    lbp = local_binary_pattern(img_proc, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(27), density=True)
    
    # 3. Simple intensity features (surprisingly useful)
    # Divide image into 4 quadrants (left eye, right eye area)
    h, w = img_proc.shape
    quadrants = [
        img_proc[0:h//2, 0:w//2],   # top-left
        img_proc[0:h//2, w//2:w],   # top-right
        img_proc[h//2:h, 0:w//2],   # bottom-left
        img_proc[h//2:h, w//2:w]    # bottom-right
    ]
    
    quad_features = []
    for quad in quadrants:
        quad_features.extend([
            quad.mean(), quad.std(), quad.max() - quad.min(),
            np.percentile(quad, 25), np.percentile(quad, 75)
        ])
    
    # 4. Edge density (eyebrow position matters)
    edges = cv2.Canny(img_proc, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 5. Horizontal intensity profile (eye openness)
    horizontal_profile = img_proc.mean(axis=1)
    h_profile_features = [
        horizontal_profile.mean(),
        horizontal_profile.std(),
        horizontal_profile.max() - horizontal_profile.min(),
        np.argmax(horizontal_profile) / h  # position of brightest row
    ]
    
    # Combine all
    features = np.concatenate([
        hog_feat, lbp_hist, quad_features, 
        [edge_density], h_profile_features
    ])
    
    return features

def load_dataset():
    """Load and prepare dataset"""
    X, y = [], []
    
    print("\n📂 Loading dataset...")
    for idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(DATA_DIR, emotion)
        if not os.path.exists(emotion_dir):
            print(f"  ❌ {emotion} folder not found!")
            continue
        
        images = [f for f in os.listdir(emotion_dir) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"  ✅ {emotion}: {len(images)} images")
        
        for img_file in tqdm(images, desc=emotion):
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            features = extract_robust_features(img)
            X.append(features)
            y.append(idx)
    
    return np.array(X), np.array(y)

def train_and_evaluate():
    """Train model and show results"""
    # Load data
    X, y = load_dataset()
    
    if len(X) == 0:
        print("\n❌ No images found! Check your dataset folder.")
        return None, None
    
    print(f"\n📊 Total samples: {len(X)}")
    print(f"📊 Feature dimension: {X.shape[1]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=RANDOM_STATE)
    }
    
    best_model = None
    best_accuracy = 0
    
    print("\n🤖 Training models...")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"  {name}: {acc:.3f} ({acc*100:.1f}%)")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_scaler = scaler
            best_name = name
    
    print(f"\n🏆 Best model: {best_name} with {best_accuracy:.3f} accuracy")
    
    # Detailed report
    y_pred = best_model.predict(X_test_scaled)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - Accuracy: {best_accuracy:.3f}')
    plt.colorbar()
    plt.xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45)
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)
    
    # Add numbers
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix_final.png')
    plt.show()
    
    # Save model
    dump(best_model, "best_emotion_final.pkl")
    dump(best_scaler, "scaler_final.pkl")
    print("\n💾 Model saved as 'best_emotion_final.pkl'")
    
    return best_model, best_scaler

def test_single_image(model, scaler):
    """Test on a single image with file picker"""
    # Create root window and hide it
    root = Tk()
    root.withdraw()
    
    # Ask user to select image
    print("\n📁 Please select an image file...")
    file_path = filedialog.askopenfilename(
        title="Select Eye Region Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_path:
        print("❌ No file selected")
        return
    
    # Load and predict
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Could not read image")
        return
    
    # Extract features
    features = extract_robust_features(img)
    features_scaled = scaler.transform([features])
    
    # Predict
    pred_idx = model.predict(features_scaled)[0]
    probs = model.predict_proba(features_scaled)[0]
    emotion = EMOTIONS[pred_idx]
    confidence = probs[pred_idx]
    
    # Display result
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"Input Image\nSize: {img.shape}")
    axes[0].axis('off')
    
    # Result
    colors = {'anger': 'red', 'happy': 'gold', 'sad': 'blue', 'surprise': 'green'}
    axes[1].axis('off')
    axes[1].text(0.5, 0.4, f"Prediction: {emotion.upper()}", 
                fontsize=24, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors.get(emotion, 'white')))
    axes[1].text(0.5, 0.7, f"Confidence: {confidence:.2f}", 
                fontsize=16, ha='center', va='center')
    
    # Add probability bars
    for i, (em, prob) in enumerate(zip(EMOTIONS, probs)):
        axes[1].bar(i, prob, label=em, color=colors.get(em, 'gray'), alpha=0.7)
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(len(EMOTIONS)))
    axes[1].set_xticklabels(EMOTIONS, rotation=45)
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Class Probabilities')
    
    plt.tight_layout()
    plt.savefig('test_result.png')
    plt.show()
    
    print(f"\n✅ Prediction: {emotion}")
    print(f"📊 Confidence: {confidence:.2f}")
    print(f"📈 All probabilities:")
    for em, prob in zip(EMOTIONS, probs):
        print(f"    {em}: {prob:.3f}")

def live_detection(model, scaler):
    """Live webcam detection"""
    print("\n🎥 Starting Live Detection")
    print("="*40)
    print("Tips for best results:")
    print("• Look directly at camera")
    print("• Ensure good lighting")
    print("• Keep face centered")
    print("• Press 'q' to quit")
    print("• Press 's' to save screenshot")
    print("="*40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    # Smoothing buffer
    pred_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Eye region (adjust these values for your face)
        eye_x = w // 4
        eye_y = h // 3
        eye_w = w // 2
        eye_h = h // 6
        
        eye_region = gray[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
        
        if eye_region.size > 0:
            # Extract features
            features = extract_robust_features(eye_region)
            features_scaled = scaler.transform([features])
            
            # Predict
            pred_idx = model.predict(features_scaled)[0]
            probs = model.predict_proba(features_scaled)[0]
            confidence = probs[pred_idx]
            emotion = EMOTIONS[pred_idx]
            
            # Smoothing (average last 5 predictions)
            pred_buffer.append((pred_idx, confidence))
            if len(pred_buffer) > 5:
                pred_buffer.pop(0)
            
            # Get most common prediction
            from collections import Counter
            smoothed_idx = Counter([p[0] for p in pred_buffer]).most_common(1)[0][0]
            smoothed_confidence = np.mean([p[1] for p in pred_buffer if p[0] == smoothed_idx])
            smoothed_emotion = EMOTIONS[smoothed_idx]
            
            # Color based on emotion
            colors = {'anger': (0, 0, 255), 'happy': (0, 255, 255), 
                     'sad': (255, 0, 0), 'surprise': (0, 255, 0)}
            color = colors.get(smoothed_emotion, (255, 255, 255))
            
            # Draw rectangle around eyes
            cv2.rectangle(frame, (eye_x, eye_y), (eye_x+eye_w, eye_y+eye_h), color, 2)
            
            # Display emotion
            cv2.putText(frame, f"{smoothed_emotion.upper()}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(frame, f"Confidence: {smoothed_confidence:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Confidence bar
            cv2.rectangle(frame, (10, 110), (10 + int(smoothed_confidence*200), 130), color, -1)
            cv2.rectangle(frame, (10, 110), (210, 130), (255, 255, 255), 1)
        
        cv2.putText(frame, "Press 'q' quit | 's' save", (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Emotion Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'emotion_capture_{smoothed_emotion}_{smoothed_confidence:.2f}.png', frame)
            print(f"📸 Saved: {smoothed_emotion} ({smoothed_confidence:.2f})")
    
    cap.release()
    cv2.destroyAllWindows()

# Main menu
if __name__ == "__main__":
    print("="*50)
    print("EMOTION DETECTION SYSTEM")
    print("="*50)
    
    while True:
        print("\n" + "="*40)
        print("MAIN MENU")
        print("="*40)
        print("1. Train new model")
        print("2. Test on single image (file picker)")
        print("3. Live webcam detection")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            model, scaler = train_and_evaluate()
            
        elif choice == '2':
            try:
                model = load("best_emotion_final.pkl")
                scaler = load("scaler_final.pkl")
                test_single_image(model, scaler)
            except:
                print("❌ No trained model found. Please train first (option 1)")
            
        elif choice == '3':
            try:
                model = load("best_emotion_final.pkl")
                scaler = load("scaler_final.pkl")
                live_detection(model, scaler)
            except:
                print("❌ No trained model found. Please train first (option 1)")
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")
