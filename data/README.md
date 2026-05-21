# Data

This directory is intentionally **not versioned**. Raw images and extracted features are not included in the repository due to size constraints and licensing restrictions.

***

## Dataset Overview

The dataset used in this project was **built from scratch**, collecting and organizing images from heterogeneous sources — both real photographs and AI-generated images produced by modern diffusion models.

| Property | Value |
|----------|-------|
| Total images (training set) | 4,424 |
| Class 0 — Real | 2,347 (53.05%) |
| Class 1 — AI-generated | 2,077 (46.95%) |
| Train / Test split | 80% / 20% (stratified) |
| Training set size | 3,539 images |
| Test set size | 885 images |
| External holdout set | 1,097 images |

The split was performed using `train_test_split` from scikit-learn with `stratify=y` to preserve class proportions in both subsets, and `random_state=42` for full reproducibility.

***

## Class Descriptions

### Class 0 — Real
Authentic photographs collected primarily from [Unsplash](https://unsplash.com), a platform offering high-resolution images under a free license. Additional images were sourced from other photography websites. The selection spans diverse subject categories — landscapes, portraits, objects, and urban scenes — to reduce the risk of the classifier learning content-related patterns instead of synthesis artifacts.

### Class 1 — AI-generated
Images produced by **three distinct text-to-image generators** based on modern diffusion architectures:

| Generator | Base Model | URL |
|-----------|-----------|-----|
| **Lucid Origin** | Leonardo.AI (custom diffusion) | https://leonardo.ai |
| **NanoBanana** | Google Gemini multimodal | https://gemini.google.com |
| **all-images.ai** | Proprietary diffusion (text-to-image) | https://all-images.ai |

Using multiple generators was a deliberate choice to reduce overfitting to a single model's visual style and improve generalization across different synthesis architectures.

***

## Feature Extraction

Raw images are not used directly as model input. Each image is converted into a **512-dimensional feature vector** using the CLIP model (`openai/clip-vit-base-patch32`) from OpenAI, kept frozen as a feature extractor (no fine-tuning).

The resulting matrix is saved as:

```
data/
└── dataset_features.csv    ← 512 numerical columns + 1 label column (0 or 1)
```

The StandardScaler fitted on the training set is saved separately:

```
data/
└── scaler.pkl              ← Serialized via joblib for inference reproducibility
```

***

## How to Reproduce the Dataset

Since raw images cannot be redistributed, follow these steps to reconstruct the dataset:

1. **Real images**: Download photographs from [Unsplash](https://unsplash.com) covering diverse subjects (landscapes, portraits, objects, urban scenes).
2. **AI-generated images**: Use any diffusion-based text-to-image tool (e.g., Stable Diffusion, DALL-E, Midjourney, Leonardo.AI) to generate synthetic images from text prompts.
3. Organize images into two folders:
   ```
   data/raw/
   ├── real/        ← Class 0
   └── ai_generated/ ← Class 1
   ```
4. Run the feature extraction notebook:
   ```
   notebooks/Tesi_Parte2ML.ipynb
   ```
   This will produce `dataset_features.csv` and `scaler.pkl`.

***

## Notes on Class Balance

The dataset presents a slight imbalance (~6 percentage points in favor of the Real class). This was considered acceptable, as it stays well below the critical 2:1 ratio that would require oversampling or class weighting techniques. Stratified splitting ensures both train and test sets reflect the original distribution identically.

***

## License

Real images sourced from Unsplash are available under the [Unsplash License](https://unsplash.com/license). AI-generated images were produced for academic research purposes only. Do **not** redistribute raw images without verifying the license of the specific source.
