# 🩺 Prostate Cancer Grading with Deep Learning (PANDA Challenge)

This repository contains an end-to-end project for **automatic prostate cancer grading** using histopathology images from the **PANDA dataset**.  
The project demonstrates **data preprocessing, augmentation, model development (ResNet50 & EfficientNet), model interpretability (Grad-CAM), and deployment with Streamlit**.

---

##  Project Pipeline

### 1. Dataset
- **Dataset**: [PANDA (Prostate cANcer graDe Assessment) Challenge](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data)  
- **Format**: Whole-slide images (WSIs) + Labels (`isup_grade`, `gleason_score`)  
- For efficiency, WSIs are converted into smaller **PNG image patches (256×256 / 512×512)**.

---

### 2. Preprocessing
- Mounted **Google Drive** in Google Colab for dataset access
- Converted WSIs to PNG format (downsampled patches)
- Resized patches to `256x256`
- Normalized pixel values `[0,1]`
- Stratified **Train/Validation/Test split**

---

### 3. Data Augmentation
Using `ImageDataGenerator` and **Albumentations**:
- Random rotations (0–180°)
- Horizontal/Vertical flips
- Random zooms
- Brightness/Color jitter
- Shear & shift transformations

This helps simulate a larger dataset and reduce overfitting.

---

### 4. Models
Two transfer learning architectures were tested:

#### 🔹 ResNet50
- Pretrained on ImageNet  
- Fine-tuned last 30 layers  
- Added custom head:
  - Global Average Pooling  
  - Dense(256, ReLU) + Dropout(0.5)  
  - Dense(6, Softmax)  

#### 🔹 EfficientNetB3
- Pretrained on ImageNet  
- Fine-tuned last 30 layers  
- Similar custom classifier head  

Both models trained with:
- Optimizer: Adam (`1e-4` then `1e-5` for fine-tuning)  
- Loss: Categorical Crossentropy  
- Metrics: Accuracy  

---

### 5. Model Training Strategy
1. **Stage 1**: Freeze base, train only top layers (feature extraction)  
2. **Stage 2**: Unfreeze last layers, train with lower learning rate (fine-tuning)  
3. **Callbacks**:
   - EarlyStopping (patience=5)  
   - ReduceLROnPlateau (factor=0.2, patience=3)  

---

### 6. Model Interpretability
To understand what the model "sees":
- Implemented **Grad-CAM** visualizations
- Highlights regions in histopathology patches that influenced predictions

---

### 7. Deployment
- Built an interactive **Streamlit app** for:
  - Uploading histopathology images  
  - Running inference (ResNet/EfficientNet)  
  - Displaying predicted ISUP grade + Grad-CAM heatmap overlay  

---

##  Results
- Experiments run on **Google Colab (Tesla T4 GPU)**  
- Trained on **3k–10k image patches** due to memory limits  
- Observations:
  - ResNet50 achieved better validation stability than EfficientNetB3 
  - Accuracy improved significantly with **data augmentation + fine-tuning**  
  - Grad-CAM provides interpretability for medical insights  

*(Final accuracy numbers will depend on dataset size & fine-tuning strategy — update here once trained fully!)*

---


