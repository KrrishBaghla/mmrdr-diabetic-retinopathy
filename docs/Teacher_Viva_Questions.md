# 🎓 Teacher Viva & Project Defense Questions: Paper 2

This guide is specifically tailored for your presentation on **"A multimodal retinal image dataset for diabetic retinopathy detection using foundation models"**. It will prepare you for questions a professor might ask a 2nd-year student implementing deep learning on medical images.

---

## 🟢 Basic Level: Medical Domain & Paper Understanding

**Q: What is the main objective of the research paper you implemented?**
*A:* The paper introduces "MMRDR," an open-source dataset containing thousands of retinal images across three modalities (CFP, UWF, and OCT). Its goal is to benchmark how well "Foundation Models" (large pre-trained AI models) can detect Diabetic Retinopathy (DR) severity and classify specific retinal lesions.

**Q: What do CFP, UWF, and OCT stand for?**
*A:* 
- **CFP (Color Fundus Photography):** Standard photographs of the back of the eye (the retina). Captures a 30-60 degree view.
- **UWF (Ultra-Widefield Fundus):** Similar to CFP but captures a much wider 200-degree view, helping spot diseases on the edges of the retina.
- **OCT (Optical Coherence Tomography):** A 3D cross-sectional scan showing the physical layers and thickness of the retina. Used heavily to find Diabetic Macular Edema (fluid build-up).

**Q: What is Diabetic Retinopathy (DR) and how does the AI grade it?**
*A:* DR is an eye disease caused by diabetes, where high blood sugar damages blood vessels in the retina. The AI grades it on a 5-point severity scale: from 0 (No DR) to 4 (Proliferative DR, the most severe stage involving new abnormal blood vessels).

---

## 🟡 Intermediate Level: Your Implementation & AI Concepts

**Q: The paper uses massive 38-billion parameter Vision-Language Models. As a 2nd year, how did you implement this?**
*A:* Training a massive model was computationally impossible on my hardware. Instead, I carefully studied the paper's *Baseline Technical Validation* section. The authors benchmarked standard ImageNet-pretrained models like **ResNet-50** against their massive models. I reproduced this baseline: I took a pre-trained ResNet-50, adapted its final layer for 5-class DR grading, and fine-tuned it on retinal images. This demonstrates Transfer Learning, which is the core concept behind all foundation models.

**Q: Why do we use a "pretrained" model instead of training from scratch?**
*A:* A model like ResNet-50, pre-trained on ImageNet, already knows how to extract basic visual features like edges, shapes, and textures. By starting with this model and "fine-tuning" it on eye images, the AI converges much faster and requires significantly fewer medical images to learn how to detect microaneurysms or hemorrhages.

**Q: Why did you resize the images to 512x512?**
*A:* I specifically followed the methodology outlined in the paper. Retinal images contain tiny diagnostic details (like microaneurysms). Resizing below 512x512 would blur these essential features and lead to misdiagnosis, while keeping them at 4K resolution would exceed GPU memory limits during training.

---

## 🔴 Advanced Level: Model Architecture & Evaluation

**Q: How does a CNN (like ResNet) learn to detect Diabetic Retinopathy?**
*A:* A Convolutional Neural Network uses mathematical filters to scan the image. The early layers learn to detect simple edges. The deeper layers combine those edges to detect complex patterns, such as the hard exudates (yellow protein leaks) or small blood spots (hemorrhages) that indicate severity in DR. The "Residual" connections in ResNet help the network pass gradients backward without them vanishing during training.

**Q: What metrics did you use to evaluate your model, and why didn't the paper use AUC?**
*A:* I used **Accuracy** and the **F1-Score**, matching the paper's methodology. Accuracy evaluates overall correct predictions. The F1-Score balances Precision and Recall, which is crucial in medical datasets where the classes might be imbalanced (e.g., more "healthy" eyes than "severe DR" eyes). The paper authors explicitly mentioned they excluded the AUC metric due to the prohibitive computational costs of extracting logits from the large Vision-Language models they were testing.

**Q: If you had more time and computing power, how would you improve this?**
*A:* I would implement multi-modal fusion. Currently, the model only looks at CFP (Color photographs) at a time. The real breakthrough of the paper is leveraging all three modalities together. I would build a multi-branch network where one branch processes the 2D fundus image and the other processes the 3D OCT scan, joining their features before the final classification layer to give a highly accurate, holistic diagnosis.

---

## 🔥 Expert Level: Your Custom Additions (Grad-CAM, Imbalance, Metrics)

**Q: Why did you implement Grad-CAM in your web application?**
*A:* Grad-CAM (Gradient-weighted Class Activation Mapping) makes the AI "explainable". In medicine, doctors cannot trust a "black box" that just spits out a diagnosis. Grad-CAM produces a visual heatmap highlighting the exact regions of the retina (like microaneurysms or fluid leaks) that caused the model to output a specific DR grade. This proves the model is looking at the actual disease and not background artifacts.

**Q: How did you address the natural class imbalance in medical datasets like MMRDR?**
*A:* Diabetic Retinopathy datasets are heavily skewed towards mild or healthy classes (Grades 0 and 1) with fewer severe cases (Grades 3 and 4). If ignored, the model would lazily predict "healthy" most of the time to achieve high baseline accuracy. I handled this during training using class weights in the loss function (penalizing the model more heavily for missing a severe case) / oversampling the minority classes to ensure the network learned the severe features equally well.

**Q: Why did you include a Confusion Matrix and Per-Class F1 scores instead of just overall Accuracy?**
*A:* Overall accuracy can be misleading in imbalanced datasets. For example, if 90% of eyes are healthy, a model that predicts "healthy" every time gets 90% accuracy but is useless medically. The Confusion Matrix explicitly shows where the model misdiagnoses (e.g., confusing Grade 2 with Grade 3). Per-class F1 ensures the model performs reliably across *all* severity levels, balancing Precision and Recall for the rare but highly dangerous severe DR cases.

**Q: What are the limitations of your project?**
*A:* Three main limitations: First, while the MMRDR dataset is multimodal, my practical implementation mostly focuses on a single modality (e.g., CFP) due to hardware constraints. Second, compared to the 38-billion parameter foundation models in the paper, my ResNet-50 trades off some deep contextual reasoning for computational efficiency. Third, this is a technical prototype and lacks formal clinical validation by ophthalmologists in a real-world setting.
