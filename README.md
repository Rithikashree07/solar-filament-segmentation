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

🏗️ Model Architecture
U-Net
The primary segmentation architecture used in this project is U-Net.
U-Net is a convolutional neural network architecture designed specifically for image segmentation.
It consists of:
- Encoder
- Bottleneck
- Decoder
- Skip connections
The encoder extracts important features from the input image, while the decoder reconstructs the spatial information required to produce a segmentation mask.
Why U-Net?
U-Net is suitable for this project because:
- It performs pixel-level segmentation.
- It preserves spatial information using skip connections.
- It is widely used for scientific and medical image segmentation.
- It can work effectively with relatively limited training datasets.
🔬 Ensemble Learning
The project explores an ensemble-based segmentation approach.
Instead of depending on a single prediction, ensemble learning can combine predictions from multiple trained models or prediction outputs to improve robustness.
The general concept is:
Model 1 ──┐
          │
Model 2 ──┼──→ Ensemble / Combined Prediction
          │
Model 3 ──┘
                    ↓
             Final Segmentation

The purpose of the ensemble approach is to reduce dependence on a single model prediction and obtain more consistent segmentation results.
🛠️ Technologies Used
Programming Language
- Python
Deep Learning
- PyTorch
Image Processing
- OpenCV
- PIL / Pillow
- NumPy
Data Processing
- Pandas
Visualization
- Matplotlib
Development Environment
- Python 3.x
- PyTorch
📁 Project Structure
solar-filament-segmentation/
│
├── model/
│   └── Solar_Filament_Best_Model (2).pth
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── prepare_masks.py
│   └── check_masks.py
│
├── model.py
├── train.py
├── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore

The exact files may vary depending on the current development version of the project.

⚙️ Dataset Preparation
The dataset preparation process includes:
1. Dataset Inspection
The original COCO-style JSON annotation file is inspected to understand:
- Images
- Annotations
- Categories
- Image IDs
- Annotation IDs
2. Mask Generation
The annotations are converted into segmentation masks.
COCO JSON Annotations
          ↓
     Annotation Parsing
          ↓
    Polygon / Mask Data
          ↓
     Binary Masks

3. Dataset Splitting
The available images are divided into:
- Training set
- Validation set
For this project:
Total images     : 707
Training images  : 565
Validation images: 142

🧩 Patch Generation
Large solar images are divided into smaller patches during preprocessing.
This helps:
- Reduce memory requirements.
- Allow the model to focus on local filament structures.
- Make training more computationally manageable.
The prepared dataset contained approximately:
Total patches      : 9040
Filament patches   : 3572
Empty patches      : 5468

🚀 Installation
Clone the repository:
git clone https://github.com/Rithikashree07/solar-filament-segmentation.git

Move into the project directory:
cd solar-filament-segmentation

Install the required dependencies:
pip install -r requirements.txt

▶️ Running the Project
Dataset Inspection
python scripts/inspect_dataset.py

Mask Preparation
python scripts/prepare_masks.py

Model Training
python train.py

Prediction
python predict.py

The exact command-line arguments may depend on the implementation in the current version of the project.

📈 Training
The model is trained using the prepared solar filament dataset.
The training process involves:
1. Loading training images.
2. Loading corresponding segmentation masks.
3. Applying preprocessing.
4. Feeding images into the U-Net model.
5. Calculating segmentation loss.
6. Performing backpropagation.
7. Updating model parameters.
8. Validating the model on validation images.
9. Saving the best-performing model.
The trained model is saved as a PyTorch .pth file.
🔍 Prediction
After training, the model can be used to generate segmentation predictions for solar images.
The prediction workflow is:
Input Solar Image
       ↓
Preprocessing
       ↓
Trained U-Net
       ↓
Predicted Mask
       ↓
Filament Segmentation

The predicted mask can then be visualized alongside the original solar image.
📊 Evaluation
The segmentation model can be evaluated using metrics such as:
- Dice Coefficient
- Intersection over Union (IoU)
- Precision
- Recall
- Pixel Accuracy
Dice Coefficient
The Dice coefficient measures the overlap between the predicted segmentation and the ground-truth segmentation.
Dice = 2 × |Prediction ∩ Ground Truth|
       --------------------------------
       |Prediction| + |Ground Truth|

IoU
Intersection over Union measures the ratio between the intersection and union of the predicted and ground-truth regions.
IoU = Intersection / Union

💻 Hardware
Deep learning segmentation training can be computationally intensive.
Recommended hardware:
- RAM: 16 GB or higher
- GPU: NVIDIA GPU with 8 GB+ VRAM recommended
- Storage: SSD recommended
- CUDA: Recommended for GPU acceleration
The project can also run on CPU, but training may take significantly longer.
📌 Current Results
The project successfully demonstrates the complete pipeline from:
Dataset
   ↓
Annotation Processing
   ↓
Mask Generation
   ↓
Preprocessing
   ↓
Model Training
   ↓
Prediction
   ↓
Solar Filament Segmentation

The trained model is available in the repository for experimentation.
🔮 Future Improvements
Future development can include:
- Training multiple segmentation architectures.
- Improving the ensemble strategy.
- Hyperparameter optimization.
- Data augmentation.
- Addressing class imbalance.
- Improving segmentation metrics.
- Testing on additional solar datasets.
- Developing a user-friendly prediction interface.
- GPU-accelerated training.
- Real-time solar image analysis.
🌞 Applications
Accurate solar filament segmentation can support:
- Solar image analysis
- Solar physics research
- Solar activity monitoring
- Automated solar feature detection
- Scientific image processing
- Space weather research
👩‍💻 Author
Rithika Shree B
M.Sc. Artificial Intelligence and Machine Learning
Coimbatore Institute of Technology
🔗 Repository
GitHub:
https://github.com/Rithikashree07/solar-filament-segmentation
📜 Disclaimer
This project is developed for academic and research purposes. The segmentation results are intended for experimentation and learning and should not be considered a substitute for expert scientific analysis.
⭐ Acknowledgement
This project uses the MAGFiLO 1.0 Kaggle 2026 dataset for solar filament segmentation research and experimentation.
