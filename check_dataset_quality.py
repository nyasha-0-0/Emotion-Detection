import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "dataset"
EMOTIONS = ['anger', 'happy', 'sad', 'surprise']

print("="*50)
print("DATASET QUALITY CHECK")
print("="*50)

for emotion in EMOTIONS:
    emotion_dir = os.path.join(DATA_DIR, emotion)
    if not os.path.exists(emotion_dir):
        print(f"{emotion}: Folder not found!")
        continue
    
    images = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"\n{emotion.upper()}: {len(images)} images")
    
    # Check a few images
    sample_images = images[:3]
    for img_file in sample_images:
        img_path = os.path.join(emotion_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"  {img_file}: size={img.shape}, min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
            
            # Display first image
            if img_file == sample_images[0]:
                plt.figure(figsize=(10, 8))
                plt.subplot(2, 2, EMOTIONS.index(emotion)+1)
                plt.imshow(img, cmap='gray')
                plt.title(f"{emotion} - {img.shape}")
                plt.axis('off')
        else:
            print(f"  {img_file}: CORRUPTED or can't read!")

plt.tight_layout()
plt.savefig('dataset_samples.png')
plt.show()

print("\n" + "="*50)
print("RECOMMENDATIONS:")
print("="*50)
print("1. Are these images cropped to show ONLY eyes?")
print("2. Are they grayscale?")
print("3. Do they have good contrast?")
print("4. Are the expressions clearly visible?")
