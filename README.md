# Advanced PCOS Detection using Deep Learning

A multimodal deep learning system for detecting **Polycystic Ovary Syndrome (PCOS)** using ultrasound imaging and clinical tabular data. The project comprises three models — a CNN-based image classifier, an MLP-based clinical data classifier, and a late-fusion multimodal model that combines both.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
  - [1. CNN Model — Ultrasound Image Classifier](#1-cnn-model--ultrasound-image-classifier)
  - [2. MLP Model — Clinical Data Classifier](#2-mlp-model--clinical-data-classifier)
  - [3. Fusion Model — Multimodal Classifier](#3-fusion-model--multimodal-classifier)
- [Results Summary](#results-summary)
- [Requirements](#requirements)
- [Project Structure](#project-structure)

---

## Overview

PCOS is a common hormonal disorder affecting people with ovaries. This project builds three complementary deep learning models to automate PCOS detection:

| Model | Input Modality | Test Accuracy |
|---|---|---|
| CNN | Ultrasound images | 99.61% |
| MLP | Clinical/tabular data | 99.25% |
| Fusion (Multimodal) | Images + Clinical data | **99.75%** |

The fusion model achieves the highest accuracy by leveraging complementary information from both imaging and clinical features.

---

## Dataset

### Ultrasound Images
- **Total images:** 15,392 ultrasound scans
- **Classes:** `infected` (PCOS positive) / `notinfected` (PCOS negative)
- **Split:** Train 70% (10,774) / Validation 15% (2,308) / Test 15% (2,310)
- **Format:** `.png`, `.jpg`, `.jpeg` at 224×224 resolution

### Clinical/Tabular Data
- **File:** `PCOS_extended_dataset.csv`
- **Records:** 2,000 rows × 44 columns
- **Target column:** `PCOS (Y/N)` — binary (0 = no PCOS, 1 = PCOS)
- **Class distribution (raw):** 1,392 non-PCOS / 608 PCOS (imbalanced → balanced via SMOTE)

**Clinical features include:** Age, BMI, weight/height, blood group, hemoglobin, pulse/RR rate, cycle regularity and length, hormone levels (FSH, LH, AMH, TSH, PRL, beta-HCG), follicle counts and sizes (left/right), skin/hair/lifestyle indicators, and blood pressure.

---

## Models

### 1. CNN Model — Ultrasound Image Classifier

**Notebook:** `CNN_model.ipynb`

#### Image Preprocessing

1. **Corrupt image scan** — filters out unreadable images before training
2. **Watershed Segmentation** — grayscale → Otsu binary threshold → morphological dilation (3×3 kernel, 3 iterations) → distance transform → sure foreground/background markers → watershed with red boundary overlay
3. **Multilevel Thresholding** — two Otsu thresholds (100, 150) applied on grayscale, regions color-mapped with JET colormap for cyst detection
4. **Data Augmentation** (training only):
   - `RandomRotation(±30°)`
   - `RandomHorizontalFlip` + `RandomVerticalFlip`
   - `RandomResizedCrop(224, scale=(0.8, 1.0))`
   - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)`
5. Resize to **224×224** and normalize to tensor

#### Architecture — `PCOS_CNN`

```
ResNet18 (pretrained, ImageNet)
  └─ fc: Linear(512 → 32)        ← feature extraction head

Classifier:
  Linear(32 → 128) → ReLU → BatchNorm1d(128) → Dropout(0.3)
  Linear(128 → 2)                ← 2-class output
```

#### Training Configuration

| Parameter | Value |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |
| LR Scheduler | StepLR (step_size=10, γ=0.1) |
| Epochs | 5 |
| Batch size | 32 |

#### Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.0357 | 98.79% | 0.0205 | 99.44% |
| 2 | 0.0165 | 99.50% | 0.0128 | 99.57% |
| 3 | 0.0146 | 99.55% | 0.0115 | 99.57% |
| 4 | 0.0143 | 99.55% | 0.0041 | 99.87% |
| 5 | 0.0195 | 99.40% | 0.0184 | 99.48% |

**Final Test Accuracy: 99.61%** (Loss: 0.0129)

**Saved model:** `pcos_detection/cnn_model.pth`

---

### 2. MLP Model — Clinical Data Classifier

**Notebook:** `MPL_FINAL_MODEL.ipynb`

#### Data Preprocessing

1. **Missing value imputation** — median for numerical columns, mode for categorical; 3 missing values in `Marraige Status (Yrs)` filled with median
2. **Column cleanup** — strip whitespace, drop non-informative columns (`Sl. No`, `Patient File No.`)
3. **Z-score normalization** (`StandardScaler`) on all numerical features
4. **Recursive Feature Elimination (RFE)** — using `RandomForestClassifier` to select the top **20 most informative features**:
   > `Weight (Kg)`, `Cycle(R/I)`, `Cycle length(days)`, `beta-HCG(mIU/mL)`, `FSH(mIU/mL)`, `LH(mIU/mL)`, `FSH/LH`, `Hip(inch)`, `TSH (mIU/L)`, `AMH(ng/mL)`, `PRL(ng/mL)`, `Vit D3 (ng/mL)`, `Weight gain(Y/N)`, `hair growth(Y/N)`, `Skin darkening (Y/N)`, `Fast food (Y/N)`, `Follicle No. (L)`, `Follicle No. (R)`, `Avg. F size (L) (mm)`, `Avg. F size (R) (mm)`
5. **SMOTE** (Synthetic Minority Oversampling) — balances classes from 1,392/608 to 1,392/1,392
6. **Train/test split** — 80/20 → 1,600 train / 400 test (stratified)
7. Convert to `float32` PyTorch tensors

#### Architecture — `PCOS_MLP`

```
Input: 20 clinical features

Linear(20 → 64)   → ReLU → BatchNorm1d(64)  → Dropout(0.3)
Linear(64 → 128)  → ReLU → BatchNorm1d(128) → Dropout(0.3)
Linear(128 → 64)  → ReLU → BatchNorm1d(64)  → Dropout(0.3)
Linear(64 → 1)    → Sigmoid                  ← binary output
```

#### Training Configuration

| Parameter | Value |
|---|---|
| Loss | BCELoss (Binary Cross-Entropy) |
| Optimizer | Adam (lr=0.001) |
| Epochs | 50 |
| Batch size | 32 |

#### Training Results (selected epochs)

| Epoch | Loss | Accuracy |
|---|---|---|
| 1 | 24.87 | 75.81% |
| 5 | 12.22 | 90.44% |
| 10 | 9.26 | 93.00% |
| 18 | 5.90 | 95.50% |
| 50 | ~6.27 | ~94.62% |

**Final Test Accuracy: 99.25%**

**Saved model:** `pcos_detection/MLP_model2.pth`

---

### 3. Fusion Model — Multimodal Classifier

**Notebook:** `fusion_model (4).ipynb`

The fusion model combines the CNN image branch and MLP tabular branch via **late fusion** — both branches independently extract 32-dimensional feature embeddings which are then concatenated and passed through a joint classifier.

#### Datasets

| Modality | Details |
|---|---|
| Clinical CSV | `PCOS_extended_dataset.csv` — 2,000 records |
| Ultrasound images | `enhanced_data/` — 15,392 images |
| Aligned split | Train: 1,600 / Test: 400 (stratified 80/20) |
| Training dist. | Non-PCOS: 1,114 / PCOS: 486 |
| Test dist. | Non-PCOS: 278 / PCOS: 122 |

The fusion model uses **11 tabular features** (a refined subset from the MLP's 20):
> `Age (yrs)`, `Weight (Kg)`, `Height(Cm)`, `BMI`, `Hb(g/dl)`, `Cycle(R/I)`, `Cycle length(days)`, `beta-HCG(mIU/mL)`, `FSH(mIU/mL)`, `LH(mIU/mL)`, `FSH/LH`

Scaling statistics (mean/std) are persisted to `scaling_values.json` for inference-time normalization.

#### Architecture — `PCOS_Multimodal`

```
┌──────────────────────────────────┐   ┌─────────────────────────────────┐
│      TABULAR BRANCH (MLP)        │   │       IMAGE BRANCH (CNN)         │
│  Input: 11 clinical features     │   │  Input: 224×224 RGB images        │
│  Linear(11→64)→ReLU→BN→Drop(0.3)│   │  ResNet18 (pretrained)            │
│  Linear(64→128)→ReLU→BN→Drop(0.3)   │    └─ fc: Linear(512→32)          │
│  Linear(128→64)→ReLU→BN→Drop(0.3)   │  Classifier:                      │
│  Linear(64→32)→Sigmoid           │   │  Linear(32→128)→ReLU→BN→Drop(0.3)│
│        Output: 32D               │   │  Linear(128→32)                  │
└──────────────────┬───────────────┘   │        Output: 32D               │
                   │                   └───────────────────┬──────────────┘
                   └────────────┬──────────────────────────┘
                                ▼
               ┌────────────────────────────────┐
               │      FUSION CLASSIFIER          │
               │  Concat(32 + 32 = 64D)          │
               │  Linear(64→128)→ReLU→BN→Drop(0.3)│
               │  Linear(128→2) → Softmax         │
               │      Output: 2 classes           │
               └────────────────────────────────┘
```

#### Training Configuration

| Parameter | Value |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |
| Epochs | 10 |
| Batch size | 32 |

#### Training Results

| Epoch | Loss | Train Accuracy |
|---|---|---|
| 1 | 0.3979 | 92.19% |
| 2 | 0.3533 | 96.25% |
| 5 | 0.3570 | 95.69% |
| 7 | 0.3364 | 97.62% |
| 9 | 0.3289 | 98.44% |
| 10 | 0.3308 | 98.44% |

**Final Test Loss: 0.3172 | Test Accuracy: 99.75%** ← highest of all three models

**Saved model:** `pcos_detection/multimodal_model_final.pth`

---

## Results Summary

| Model | Modality | Architecture | Test Accuracy | Saved Model |
|---|---|---|---|---|
| CNN | Ultrasound images | ResNet18 + FC | 99.61% | `cnn_model.pth` |
| MLP | Clinical tabular (20 features) | 4-layer MLP | 99.25% | `MLP_model2.pth` |
| **Fusion** | **Images + Clinical (11 features)** | **MLP + CNN + FC** | **99.75%** | `multimodal_model_final.pth` |

The multimodal fusion model consistently outperforms both unimodal models, demonstrating that combining imaging and clinical data captures complementary diagnostic signals.

---

## Requirements

```
torch
torchvision
opencv-python (cv2)
Pillow
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
openpyxl
```

Install all dependencies:

```bash
pip install torch torchvision opencv-python Pillow pandas numpy matplotlib seaborn scikit-learn imbalanced-learn openpyxl
```

---

## Project Structure

```
├── CNN_model.ipynb               # CNN image classifier (ResNet18-based)
├── MPL_FINAL_MODEL.ipynb         # MLP clinical data classifier
├── fusion_model (4).ipynb        # Multimodal fusion model
└── README.md                     # This file

# Generated at runtime (Google Drive):
pcos_detection/
├── 1/content/data/enhanced_data/ # Ultrasound images (infected / notinfected)
├── PCOS_extended_dataset.csv     # Clinical tabular dataset
├── cnn_model.pth                 # Saved CNN model weights
├── MLP_model2.pth                # Saved MLP model weights
├── multimodal_model_final.pth    # Saved fusion model weights
└── scaling_values.json           # Normalization parameters for inference
```
