# ☀️ Solar Filament Segmentation Using Ensemble Learning

## 📌 Project Overview

Solar filaments are elongated structures present in the solar atmosphere that appear as dark features in H-alpha solar images. Accurate identification and segmentation of these structures can support solar image analysis and research in solar physics.

This project focuses on the **automatic segmentation of solar filaments from H-alpha solar images** using deep learning and an ensemble-based approach.

The project uses the **MAGFiLO 1.0 Kaggle 2026 dataset**, which contains solar H-alpha images with corresponding annotations.

---

## 🎯 Objectives

- Automatically identify solar filaments from H-alpha solar images.
- Generate pixel-level segmentation masks for detected filaments.
- Prepare and process COCO-style annotations.
- Train a deep learning segmentation model.
- Evaluate the segmentation results using appropriate metrics.
- Explore ensemble learning to improve the robustness of filament segmentation.
- Generate visual predictions for unseen solar images.

---

## 📊 Dataset

### MAGFiLO 1.0 Kaggle 2026

The project uses the MAGFiLO dataset containing H-alpha solar images and annotations.

### Dataset Characteristics

- **Image type:** H-alpha solar images
- **Image format:** JPEG
- **Original image resolution:** 2048 × 2048
- **Annotation format:** COCO-style JSON
- **Training images available:** 707
- **Training split:** 565 images
- **Validation split:** 142 images

The dataset contains annotations associated with different filament identification categories.

### Annotation Categories

| ID | Category |
|---|---|
| 1 | Left |
| 2 | Right |
| 3 | Unidentifiable |
| 4 | Ambiguous |

The annotations were converted into pixel-level masks for training the segmentation model.

> **Note:** The original dataset images and annotations are not included in this repository because of their size and dataset distribution restrictions.

---

## 🧠 Methodology

The overall workflow of the project is:

```text
Solar H-alpha Images
        ↓
Dataset Inspection
        ↓
COCO Annotation Processing
        ↓
Annotation → Segmentation Masks
        ↓
Train / Validation Split
        ↓
Image Preprocessing
        ↓
Patch Generation
        ↓
Deep Learning Segmentation
        ↓
Model Training
        ↓
Prediction
        ↓
Segmentation Evaluation
        ↓
Ensemble-based Analysis
