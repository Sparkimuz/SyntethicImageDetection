# SyntethicImageDetection
Binary classifier for AI-generated image detection using CLIP feature extraction + SVM, Random Forest, MLP and Voting Ensemble — 96.72% accuracy on custom dataset.

**Bachelor's Thesis** — Università degli Studi Roma Tre  
Department of Civil, Computer Science and Aeronautical Technologies Engineering  
Academic Year 2024/2025 | Supervisor: Prof. Giuseppe Sansonetti

---

## 📌 Overview

This project investigates whether **classical Machine Learning approaches**, when combined with modern semantic feature extraction via **CLIP (Contrastive Language-Image Pre-Training)**, can achieve competitive results against end-to-end deep learning architectures in the **Fake Image Detection** domain.

The core pipeline:
1. Extract 512-dimensional visual feature vectors from images using OpenAI's CLIP (`openai/clip-vit-base-patch32`)
2. Train and evaluate three classifiers: **SVM (RBF kernel)**, **Random Forest**, and **MLP**
3. Combine them into a **Soft Voting Ensemble** for improved robustness

**Best result: 96.72% accuracy** with the Voting Ensemble on a custom-built dataset of 4,424 images.

---

## 🗂️ Project Structure

```
fake-image-detection/
│
├── notebooks/
│   ├── Tesi_Parte2ML.ipynb           # ML classifiers training & evaluation
│   ├── Tesi_DeepLearning.ipynb       # Deep learning comparison experiments
│   └── Tesi_Demo_Riconoscimento.ipynb # Interactive Gradio demo
│
├── data/
│   └── README.md                     # Dataset description and download instructions
│
├── models/
│   └── scaler.pkl                    # Saved StandardScaler for inference
│
├── results/
│   ├── confusion_matrix.png          # Confusion matrix – Voting Ensemble
│   └── model_comparison.png          # Performance comparison chart
│
├── requirements.txt
└── README.md
```

---

## 🧠 Methodology

### Feature Extraction with CLIP
Each image is processed by the frozen CLIP Vision Transformer (`ViT-B/32`), producing a **512-dimensional semantic embedding**. No fine-tuning is performed — CLIP is used purely as a feature extractor, making the pipeline computationally lightweight and reproducible without high-end GPU resources.

### Classifiers
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM (RBF kernel) | 96.63% | 94.43% | 99.09% | 96.70% |
| Random Forest | 95.53% | 93.68% | 97.62% | 95.61% |
| MLP | 96.63% | 94.74% | 98.72% | 96.69% |
| **Voting Ensemble** | **96.72%** | **94.59%** | **99.09%** | **96.79%** |

Hyperparameters were optimized via **GridSearchCV with 5-fold cross-validation** on the training set.

### Dataset
A **custom dataset** of 4,424 images was built from scratch:
- **Class 0 – Real** (2,347 images, 53.05%): High-resolution photographs from Unsplash and other photography platforms
- **Class 1 – AI-generated** (2,077 images, 46.95%): Images produced by three distinct diffusion-based generators — LucidOrigin (Leonardo.AI), NanoBanana (Google Gemini), and all-images.ai

The use of multiple generators was deliberate, aiming to prevent the classifier from overfitting to the artifacts of a single model. The dataset was split 80/20 (train/test) with stratification, plus an additional **external holdout set of 1,097 images** for out-of-distribution generalization testing.

---

## ⚙️ Setup & Usage

### Requirements
```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers` (Hugging Face)
- `scikit-learn`
- `torch`
- `Pillow`
- `gradio`
- `pandas`, `numpy`, `matplotlib`

### Run on Google Colab
Each notebook is designed to run on **Google Colab** (free tier compatible). Open any notebook and follow the step-by-step cells:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### Inference with the Gradio Demo
Launch the interactive demo (notebook `Tesi_Demo_Riconoscimento.ipynb`) to classify arbitrary images in real time. The interface displays the predicted class and confidence score.

---

## 📊 Results

The **Voting Ensemble** achieved the best overall performance, combining the complementary decision boundaries of SVM, Random Forest, and MLP via soft voting (probability averaging). It showed more uniform error distribution compared to individual models, which tend to concentrate mistakes on specific image subtypes.

On the **external holdout set** (out-of-distribution images from unseen generators), the ensemble maintained competitive discriminative capability, though with a slight accuracy drop consistent with published literature on cross-generator generalization.

---

## 🔬 Key Findings

- Classical ML classifiers, when fed with CLIP semantic embeddings, can reach **near state-of-the-art accuracy** with a fraction of the computational cost of end-to-end deep learning
- The **512-dimensional CLIP features** capture subtle statistical differences between real and AI-generated images related to the generative process itself, not just the visual style of a specific generator
- The system runs **inference in near real-time** on CPU, making it suitable for scalable content moderation pipelines
- The **"grey zone"** (decision scores between 0.40–0.60) is populated by high-quality synthetic images and atypical real photos — frequency-domain features (FFT, LBP) could further reduce this ambiguity

---

## 🚧 Limitations & Future Work

- **Cross-generator generalization**: Performance degrades on images from generators not represented in the training set
- **Dataset size**: A larger, more diverse dataset would improve robustness
- **Feature fusion**: Combining CLIP embeddings with FFT spectral features and Local Binary Patterns (LBP) could improve detection of subtle artifacts
- **Cross-generator training**: Training explicitly on diverse generators to maximize out-of-distribution generalization

---

## 📄 License

This project is released for academic and research purposes.  
If you use this work, please cite the original thesis.

---

## 👤 Author

**Tommaso Cenciotti**  
Bachelor's in Computer Engineering — Roma Tre University  
📧 t.cenciotti@gmail.com  
🔗 [GitHub](https://github.com/Sparkimuz)
