# 🎭 Emotion Detection using Machine Learning and Computer Vision

## 📌 Overview
This project focuses on detecting human emotions (happy, sad, surprise, anger) from facial images using traditional machine learning and computer vision techniques.

## ⚙️ Approach
Instead of using deep learning, this project relies on feature extraction methods such as:
- 📊 HOG (Histogram of Oriented Gradients)
- 🧩 LBP (Local Binary Patterns)
- 🌊 Gabor filters
- 😊 Facial landmark-based features (distances between eyebrows, eyes, etc.)

Facial landmarks were extracted using MediaPipe, and image preprocessing was done using OpenCV.

## 🤖 Models Used
We trained and compared multiple machine learning models:
- SVM (Support Vector Machine)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Random Forest

## 🛠️ Tools & Libraries
- 🐍 Python
- 👁️ OpenCV
- 🧠 MediaPipe
- 🔢 NumPy, Pandas
- 📈 Matplotlib, Seaborn
- ⚡ scikit-learn

## 📊 Dataset
The dataset was created by collecting images from multiple sources.  
Each emotion class had around 500 images, with efforts made to include diversity in:
- 🌍 Ethnicity
- 💡 Lighting conditions
- 🙂 Facial variations

## 🚧 Challenges
- Lack of a well-balanced dataset initially  
- Subtle differences between emotions made classification difficult  
- Handcrafted features struggled to capture complex patterns  
- Data collection and cleaning required significant effort  

## 📈 Results
- 🔄 The project is still under active development  
- 🧪 Ongoing experiments are being conducted to improve performance  

## 🚀 Status
⚠️ Work in Progress  
This project was developed as part of a team. The current repository may not contain the complete implementation, and the full codebase will be uploaded soon.

## 🧠 Learnings
- Data quality and balance are critical for emotion detection  
- Traditional ML approaches have limitations for complex visual tasks  
- Feature engineering plays a major role in model performance
