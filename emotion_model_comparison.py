import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
DATA_DIR = "dataset"
IMG_SIZE = (64, 64)
EMOTIONS = ['anger', 'happy', 'sad', 'surprise']
RANDOM_STATE = 42

# Feature extraction
def extract_features(img_gray):
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    
    # HOG features
    hog_feat = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False, feature_vector=True)
    
    # LBP features
    lbp = local_binary_pattern(img_resized, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(27), density=True)
    
    # Statistical features
    edges = cv2.Canny(img_resized, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    features = np.concatenate([hog_feat, lbp_hist, [img_resized.mean(), img_resized.std(), edge_density]])
    return features

# Load dataset
def load_data():
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
            
            features = extract_features(img)
            X.append(features)
            y.append(label_idx)
    
    return np.array(X), np.array(y)

# Train and evaluate multiple models
def train_all_models(X_train, y_train, X_test, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE),
        'SVM (Linear)': SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        predictions[name] = y_pred
        print(f"{name} Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return results, predictions

# Plot comparison bar chart
def plot_model_comparison(results):
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    scores = list(results.values())
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4A9C', '#FF6B6B']
    bars = plt.bar(models, scores, color=colors[:len(models)])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Comparison for Emotion Detection from Eye Regions', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()
    print("\n✅ Model comparison chart saved as 'model_comparison.png'")

# Plot confusion matrices
def plot_confusion_matrices(predictions, y_test, top_n=3):
    accuracies = {name: accuracy_score(y_test, pred) for name, pred in predictions.items()}
    top_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    fig, axes = plt.subplots(1, top_n, figsize=(5*top_n, 4))
    if top_n == 1:
        axes = [axes]
    
    for idx, (model_name, _) in enumerate(top_models):
        cm = confusion_matrix(y_test, predictions[model_name])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)
        disp.plot(ax=axes[idx], cmap='Blues')
        axes[idx].set_title(f'{model_name}\nAccuracy: {accuracies[model_name]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_top_models.png', dpi=150)
    plt.show()

# Cross-validation comparison - FIXED VERSION
def compare_cross_validation(X, y):
    print("\n🔄 Performing 5-fold cross-validation...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        'Naive Bayes': GaussianNB()
    }
    
    # Scale features for cross-validation
    print("  Scaling features for CV...")
    scaler_cv = StandardScaler()
    X_scaled = scaler_cv.fit_transform(X)
    
    cv_scores = {}
    for name, model in models.items():
        print(f"  Running CV for {name}...")
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        cv_scores[name] = scores
        print(f"    {name:15} CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Plot CV comparison
    plt.figure(figsize=(10, 6))
    positions = range(len(cv_scores))
    plt.boxplot([scores for scores in cv_scores.values()], positions=positions, widths=0.6)
    plt.xticks(positions, cv_scores.keys(), rotation=45, ha='right')
    plt.ylabel('Cross-Validation Accuracy', fontsize=12)
    plt.title('5-Fold Cross-Validation Comparison', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cross_validation_comparison.png', dpi=150)
    plt.show()
    
    return cv_scores

# Live detection
def live_detection(model, scaler, model_name):
    print(f"\n🎥 Starting Live Detection using {model_name}")
    print("Press 'q' to quit, 's' to save current frame")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi_x, roi_y = w // 3, h // 4
        roi_w, roi_h = w // 3, h // 6
        
        eye_region = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        if eye_region.size > 0:
            features = extract_features(eye_region)
            features_scaled = scaler.transform([features])
            pred_idx = model.predict(features_scaled)[0]
            emotion = EMOTIONS[pred_idx]
            
            colors = {'anger': (0, 0, 255), 'happy': (0, 255, 255), 
                     'sad': (255, 0, 0), 'surprise': (0, 255, 0)}
            color = colors.get(emotion, (255, 255, 255))
            
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), color, 2)
            cv2.putText(frame, f"Emotion: {emotion.upper()}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Model: {model_name}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, "Press 'q' to quit | 's' to save", (10, frame.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Emotion Detection - Eye Region Analysis', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'captured_emotion_{emotion}.png', frame)
            print(f"Saved captured frame")
    
    cap.release()
    cv2.destroyAllWindows()

# Test single image
def test_single_image(model, scaler, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image")
        return None
    
    img_resized = cv2.resize(img, IMG_SIZE)
    features = extract_features(img_resized)
    features_scaled = scaler.transform([features])
    pred_idx = model.predict(features_scaled)[0]
    emotion = EMOTIONS[pred_idx]
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized, cmap='gray')
    plt.title('Input Eye Region')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    colors = {'anger': 'red', 'happy': 'gold', 'sad': 'blue', 'surprise': 'green'}
    plt.text(0.5, 0.5, f'Predicted: {emotion.upper()}', 
            fontsize=20, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors.get(emotion, 'white')))
    plt.axis('off')
    plt.title('Prediction Result')
    
    plt.tight_layout()
    plt.savefig('test_prediction_result.png', dpi=150)
    plt.show()
    
    print(f"✅ Prediction: {emotion}")
    return emotion

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("="*70)
    print("EMOTION DETECTION FROM EYE REGIONS - MODEL COMPARISON")
    print("="*70)
    
    # Load dataset
    print("\n📂 Loading dataset and extracting features...")
    X, y = load_data()
    print(f"\n✅ Loaded {len(X)} samples with {X.shape[1]} features each")
    
    if len(X) == 0:
        print("❌ No images found!")
        exit()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=RANDOM_STATE, 
                                                        stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train all models
    print("\n🤖 Training multiple models...")
    results, predictions = train_all_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Plot comparison
    print("\n📊 Generating comparison charts...")
    plot_model_comparison(results)
    plot_confusion_matrices(predictions, y_test)
    
    # Cross-validation - USING X (original features)
    cv_scores = compare_cross_validation(X, y)
    
    # Get best model
    best_model_name = max(results, key=results.get)
    best_model = None
    
    if best_model_name == 'Random Forest':
        best_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE)
    elif 'SVM' in best_model_name:
        kernel = 'rbf' if 'RBF' in best_model_name else 'linear'
        best_model = SVC(kernel=kernel, C=1.0, random_state=RANDOM_STATE)
    elif 'KNN' in best_model_name:
        k = int(best_model_name.split('k=')[1].split(')')[0])
        best_model = KNeighborsClassifier(n_neighbors=k)
    elif 'Decision Tree' in best_model_name:
        best_model = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
    else:
        best_model = GaussianNB()
    
    best_model.fit(X_train_scaled, y_train)
    
    print(f"\n🏆 Best Model: {best_model_name} with accuracy {results[best_model_name]:.3f}")
    
    # Save best model
    dump(best_model, "best_emotion_model.pkl")
    dump(scaler, "emotion_scaler.pkl")
    print("💾 Best model saved as 'best_emotion_model.pkl'")
    
    # Interactive menu
    while True:
        print("\n" + "="*50)
        print("WHAT WOULD YOU LIKE TO DO?")
        print("="*50)
        print("1. Test on a single image from dataset")
        print("2. Start live webcam detection")
        print("3. Test on custom image path")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            import random
            test_emotion = random.choice(EMOTIONS)
            test_images = os.listdir(f"{DATA_DIR}/{test_emotion}")
            if test_images:
                test_img = random.choice(test_images)
                test_single_image(best_model, scaler, f"{DATA_DIR}/{test_emotion}/{test_img}")
            else:
                print(f"No images found")
            
        elif choice == '2':
            live_detection(best_model, scaler, best_model_name)
            
        elif choice == '3':
            img_path = input("Enter full image path: ").strip()
            if os.path.exists(img_path):
                test_single_image(best_model, scaler, img_path)
            else:
                print("❌ Image path not found!")
                
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice!")
