# 🎓 Comprehensive Internship Progress Report: Diabetic Retinopathy Detection

## 1. Project Overview & Objective
This project focuses on **Diabetic Retinopathy (DR) detection** using the **MMRDR** (Multimodal Retinal Image Dataset). The primary objective is to build and evaluate deep learning models capable of grading DR severity and classifying specific retinal lesions across different modalities.

> [!NOTE]
> The dataset leverages three key modalities:
> - **CFP (Color Fundus Photography)**: Standard 30-60 degree view.
> - **UWF (Ultra-Widefield Fundus)**: Wider 200-degree view to spot peripheral diseases.
> - **OCT (Optical Coherence Tomography)**: 3D cross-sectional scans highlighting physical retinal layers.

---

## 2. Directory & File Structure Analysis

### 📁 Directories
- **`MMRDR/`**: The core dataset folder containing three subdirectories (`MMRDR-CFP`, `MMRDR-OCT`, `MMRDR-UWF`), categorizing images by their respective medical imaging modality.
- **`findings/`**: Stores Jupyter notebooks (e.g., `train_retfound.ipynb`, `custom_architecture_resnet.ipynb`) and saved model weights (`best_multiscale_resnet.pth`, `best_retfound_cfp.pth`). These large `.pth` files represent the successfully trained baseline models.
- **`relavant pics and models/`**: Contains essential visual assets, Grad-CAM heatmaps, model result charts, and deployment code (`app.py`). This directory acts as the portfolio of visual evidence for the project's success.
- **`zip_files/`**: Contains `MMRDR_Full.zip` (18.6 GB), which is the complete, compressed archive of the multimodal dataset.
- **`Diabetic_env/`**: The local Python virtual environment encapsulating all project dependencies.

### 📄 Core Source Files
- [alternate_architechture.py](file:///c:/Users/HP/OneDrive/Desktop/BHU%20Intern/diabetes_project/paper_2/alternate_architechture.py): Implements a custom **Multi-Scale Lesion-Attention Net (MS-LAN)** based on ResNet-50. It uses a dual-branch architecture: one preserves 256x256 micro details (like microaneurysms) and the other extracts global macro context. It incorporates a `ClassBalancedFocalLoss` to combat medical data imbalance.
- [train_gnn.py](file:///c:/Users/HP/OneDrive/Desktop/BHU%20Intern/diabetes_project/paper_2/train_gnn.py): Script to facilitate training using Graph Neural Networks.
- [Teacher_Viva_Questions_Paper2.md](file:///c:/Users/HP/OneDrive/Desktop/BHU%20Intern/diabetes_project/paper_2/Teacher_Viva_Questions_Paper2.md): A vital reference document explaining the project methodology, the reasoning behind image resizing (512x512 to preserve tiny diagnostic details), and evaluation metrics (Accuracy and F1-Score).
- **`app.py`** (inside `relavant pics and models/`): The web application backend that integrates the trained models with Grad-CAM to provide visual, explainable diagnostics to doctors.

---

## 3. Key Findings & Methodology

> [!TIP]
> **Handling Class Imbalance**: Medical datasets heavily skew towards healthy eyes. The model successfully uses Class-Balanced Focal Loss and custom class weighting to ensure severe DR cases are not ignored by the network.

**Transfer Learning & Fine-Tuning**
Due to computational limits on running massive 38B parameter Foundation Models, the project successfully reproduced baseline technical validations by fine-tuning **ResNet-50**. By retaining an input size of 512x512, the architecture successfully avoids blurring out crucial microaneurysms.

**Explainable AI (Grad-CAM)**
The addition of Grad-CAM (Gradient-weighted Class Activation Mapping) is a major finding. It proves the AI is not acting as a "black box" but is actively looking at specific pathological regions (like fluid leaks or hemorrhages) to make its diagnosis.

---

## 4. Visual Evidence and Results

Below are the key visual outcomes extracted from the `relavant pics and models` folder:

### Model Architecture
The custom dual-branch architecture allows the model to process localized micro-lesions while retaining global structural context.
![Custom ResNet Architecture](relavant%20pics%20and%20models/custom_resnet_architecture.jpeg)

### Explainability with Heatmaps
Grad-CAM heatmaps successfully highlight the regions of interest the model uses to determine the severity grade.
![Custom Architecture Visuals with Heatmap](relavant%20pics%20and%20models/custom_architecture_resnet_visuals_with_heatmap.png)

### UWF Modality Results
Performance of the models when trained and validated on Ultra-Widefield Fundus images.
![UWF Results](relavant%20pics%20and%20models/UWF_results.png)

### Dataset Samples (UWF)
Example of the Ultra-Widefield modality dataset.
![UWF Dataset](relavant%20pics%20and%20models/UWF_dataset.png)

---

## 5. Conclusion & Limitations

> [!IMPORTANT]
> **Project Limitations**: 
> 1. Hardware constraints limited the full multi-modal fusion (combining CFP and OCT simultaneously). 
> 2. The ResNet-50 trades off some deep contextual reasoning found in massive Foundation Models for computational efficiency.

Overall, the project acts as a robust, explainable prototype for Diabetic Retinopathy grading. It effectively manages memory limits, addresses class imbalance, and provides visual, doctor-friendly diagnostics through its Grad-CAM integrated web application.
