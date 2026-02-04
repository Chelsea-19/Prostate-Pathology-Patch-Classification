# Prostate Cancer Gleason Patch Classification with UNI

This repository contains the implementation and experimental results for **multi-class Gleason grade classification** on prostate cancer histopathology patches using the **pathology foundation model UNI**.  
The project explores a transfer learning framework under **data imbalance and limited sample size**, with a focus on clinically critical Gleason grades (G3–G5).

This work was completed as part of an MSc-level coursework project in biomedical data science.

---

## 1. Project Overview

Prostate cancer grading using the **Gleason system** is a core clinical task but suffers from:
- high inter-observer variability,
- heavy workload for pathologists,
- severe class imbalance, especially for high-risk Gleason 5 cases.

To address these challenges, this project:
- adopts **UNI**, a domain-specific pathology foundation model pretrained on large-scale histopathology data,
- applies **weighted random sampling** to mitigate class imbalance,
- evaluates classification performance and interpretability at the patch level.

The full methodology and results are described in the accompanying report :contentReference[oaicite:0]{index=0}.

---

## 2. Dataset

- **Source**: H&E-stained prostate pathology patches from National University Hospital (NUH)
- **Classes (5)**:
  - Stroma  
  - Normal  
  - Gleason 3 (G3)  
  - Gleason 4 (G4)  
  - Gleason 5 (G5)

- **Training set**: 7,400 patches  
- **Validation set**: 2,000 patches  
- **Key challenge**: strong class imbalance, especially for G5

---

## 3. Methodology

### 3.1 Feature Extraction with UNI

- UNI is used as a **frozen backbone** for feature extraction.
- Input patches are resized to **224 × 224** to match ViT requirements.
- The backbone outputs **1024-dimensional embeddings** per patch.
- No fine-tuning is applied to the UNI backbone.

### 3.2 Classification Head

A lightweight classifier is trained on top of UNI embeddings:
- Layer normalization
- Fully connected layer (1024 → 512)
- ReLU activation
- Dropout (p = 0.3)
- Final linear layer with Softmax over 5 classes

This design targets the **fuzzy morphological boundaries** between adjacent Gleason grades.

### 3.3 Imbalanced Data Handling

To address data imbalance:
- **Weighted Random Sampling** is applied during training.
- Class weights are computed as the inverse of class frequency.
- This strategy ensures balanced batch composition without modifying the loss function.

### 3.4 Training Strategy

- Optimizer: AdamW  
- Initial learning rate: 1e-4  
- Scheduler: Cosine annealing  
- Loss function: Cross-entropy  
- Framework: PyTorch  
- Hardware: NVIDIA A100 GPU (Google Colab)

---

## 4. Results

### 4.1 Overall Performance

| Metric | Value |
|------|------|
| Accuracy | 0.9415 |
| Macro F1 | 0.9411 |
| Weighted F1 | 0.9411 |

- Strong performance across benign and malignant classes
- Gleason 3 and Gleason 5 both achieve F1-scores above 0.90
- Most errors occur between **adjacent grades (G3–G4, G4–G5)**

### 4.2 Ablation Studies

- **Weighted sampling** consistently improves macro recall and macro F1
- Learning rate sensitivity is observed, indicating the importance of careful optimization

### 4.3 Interpretability

Class-specific activation visualizations show that:
- the model focuses on **glandular morphology and cellular density**,
- learned representations align with known pathological criteria,
- errors are largely attributable to intrinsic histological ambiguity.

---
