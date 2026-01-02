# From Oversampling to DCGAN: A Systematic Comparison of Data Synthesis for Imbalanced Supervised Classification in Additive Manufacturing

This repository provides the implementation and experimental pipeline for a systematic comparison of **data synthesis methods**—ranging from traditional oversampling techniques to GAN-based generative models—for **imbalanced defect classification in additive manufacturing (AM)**.

The project investigates how the *quality of synthesized minority-class images* affects the performance of supervised CNN classifiers under extreme class imbalance.

---

## Overview

In real-world additive manufacturing processes, defect samples are extremely scarce compared to normal samples due to high inspection costs and low defect occurrence rates. This severe **class imbalance** often leads supervised classifiers to be biased toward the majority class, resulting in poor recall and unreliable defect detection.

To address this issue, this project compares:

- **Traditional oversampling methods**
  - SMOTE
  - Borderline-SMOTE
  - ADASYN
- **Generative models**
  - Fully-connected GAN
  - Deep Convolutional GAN (DCGAN)

All methods are evaluated under the **same CNN classifier architecture and evaluation metrics**, enabling a fair and controlled comparison.

---

## Dataset

The dataset consists of **high-resolution surface images captured from a fused filament fabrication (FFF) 3D printing process**, including:

- **Normal**: defect-free surfaces
- **Underfill_FR (Type 1)**: filament under-extrusion causing sparse deposition and reduced surface density
<img width="448" height="337" alt="image" src="https://github.com/user-attachments/assets/99270e59-3b87-45fb-a1ae-7401d5a1821d" />

- **Underfill_Fan (Type 2)**: excessive cooling causing discontinuous or fractured deposition patterns
<img width="449" height="338" alt="image" src="https://github.com/user-attachments/assets/13635829-52e2-40bd-9afe-73bc6378f119" />

### Dataset Characteristics

|           |	Given	      |+Augmentation |  Train	| Test |
|-----------|-------------|--------------|--------|------|
| Normal	  | 305        	| 610          |	500   |	100  |
|FR50(type1)|	197         |	394	         | 25     |	100  |
|Fan(type2) |	153	        | 306	         | 25	    | 100  |

- Original dataset size:
  - Normal: 305 images
  - Type 1 (Underfill_FR): 197 images
  - Type 2 (Underfill_Fan): 153 images
- Extreme imbalance ratio:
  - Minority-to-majority ratio ≤ 0.1
- Data augmentation:
  - Horizontal flipping applied to increase diversity
- Final train/test split:
  - Training: balanced via synthesis methods
  - Testing: 100 images per class (held-out)

---

## Methodology

### 1. Baseline CNN Classifier

A supervised **multi-class CNN classifier** is used as the baseline and evaluation backbone.

<img width="448" height="244" alt="image" src="https://github.com/user-attachments/assets/a75884e9-58cd-426f-acec-cf8bdb0c0709" />

- Input: AM surface images
- Output classes: Normal / Type 1 / Type 2
- Architecture:
  - 3 convolutional layers
  - 2 fully connected layers
- Purpose:
  - Establish baseline performance
  - Measure classification improvement after data synthesis

---

### 2. Oversampling-Based Data Synthesis

Traditional oversampling methods generate new samples via **interpolation in feature or pixel space**:

- **SMOTE**: interpolates between minority samples and their nearest neighbors
- **Borderline-SMOTE**: focuses synthesis on decision boundary samples
- **ADASYN**: adaptively generates more samples for hard-to-learn regions

While computationally efficient, these methods often produce images with **blurred or overlapping features**, which can dilute defect-specific characteristics in high-resolution AM images.

---

### 3. GAN-Based Data Synthesis

#### Fully-Connected GAN

A vanilla GAN with multilayer perceptron (MLP) architecture is first applied:

<img width="447" height="298" alt="image" src="https://github.com/user-attachments/assets/4b542986-a49f-4590-a5ad-c90843091f37" />

- Generator:
  - Maps latent vectors to image space using fully connected layers
- Discriminator:
  - Classifies real vs. generated images

**Observed limitations**:
- Severe noise in generated images
- Mode collapse
- Failure to capture spatial defect patterns
- Poor image quality even under full-shot training

Due to these limitations, GAN-generated samples were not used for classifier retraining.

---

#### Deep Convolutional GAN (DCGAN)

To address resolution and spatial representation issues, **DCGAN** is applied:

<img width="448" height="501" alt="image" src="https://github.com/user-attachments/assets/98da3dbb-b325-4e84-91ae-4bfa409c980a" />

Key architectural improvements:
- Convolutional and transposed convolutional layers
- No pooling layers
- Batch normalization for stable training
- ReLU (Generator) and LeakyReLU (Discriminator)
- Tanh output for image normalization

DCGAN enables:
- Better preservation of texture, contrast, and surface discontinuities
- More realistic defect representations despite limited diversity

---

## Experiments

### Evaluation Protocol

1. Train baseline CNN on imbalanced dataset
2. Apply data synthesis to balance minority classes
3. Retrain CNN with synthesized datasets
4. Evaluate using:
   - Precision
   - Recall
   - F1-score

---

### Quantitative Results

| Method      | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| Baseline    | 0.8342    | 0.7467 | 0.7369   |
| SMOTE       | 0.8561    | 0.7467 | 0.7206   |
| B-SMOTE     | 0.8593    | 0.7567 | 0.7417   |
| ADASYN      | 0.8619    | 0.8000 | 0.7797   |
| **DCGAN**   | **0.9048**| **0.8667** | **0.8611** |

DCGAN achieves the largest improvement across all metrics, particularly in recall.

---

### Qualitative Observations

- **Oversampling methods**
  - Produce interpolated images with feature overlap
  - Blur defect-specific textures
  - Limited improvement for visually subtle defects
- **DCGAN**
  - Captures key defect cues:
    - Contrast gaps in Underfill_FR
    - Blurry, discontinuous patterns in Underfill_Fan
  - Limited diversity due to mode collapse
  - Still more effective for classification than interpolation-based synthesis

---

## Key Findings

1. Fully-connected GANs are **structurally unsuitable** for high-resolution AM surface image synthesis.
2. Traditional oversampling methods provide a **stable baseline** under extreme data scarcity.
3. DCGAN significantly improves classification performance by **preserving defect-discriminative visual patterns**, even with limited sample diversity.
4. Image quality and spatial representation are **more critical than sheer sample count** in AM defect synthesis.

---

## Future Work

- Conditional DCGAN for improved diversity
- WGAN / WGAN-GP for training stability
- Diffusion-based models for higher fidelity synthesis
- Extension to other manufacturing defect domains

---
