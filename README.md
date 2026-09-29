# Schizophrenia Detection Using EEG Brain Signals

A deep-learning framework for **schizophrenia classification from electroencephalography (EEG) brain signals**.

The system processes EEG recordings from two groups:

* **Control / Healthy Control**
* **Schizophrenia**

The project evaluates multiple EEG-specific deep-learning architectures and performs subject-level cross-validation, hyperparameter selection, model training, testing, performance evaluation, and model interpretation using saliency maps and EEG frequency-domain analysis.

> **Important:** This project is intended for research and educational purposes. It is **not a clinically validated diagnostic system** and predictions should not be interpreted as a medical diagnosis.

---

## Overview

Schizophrenia is a complex psychiatric disorder for which EEG signals can provide information about brain activity. Machine-learning methods can be used to investigate whether patterns in EEG recordings can distinguish schizophrenia subjects from healthy/control subjects.

This project implements an end-to-end EEG classification pipeline:

```text
Raw EEG
   │
   ▼
Preprocessing
   │
   ├── Channel selection
   ├── Band-pass filtering
   └── Downsampling
   │
   ▼
EEG Segmentation
   │
   └── 5-second windows
   │
   ▼
Subject-wise Normalization
   │
   ▼
Pilot Training
   │
   └── Batch size + Learning-rate search
   │
   ▼
5-Fold Subject-level Training
   │
   ▼
Model Testing
   │
   ├── Segment-level accuracy
   ├── Subject-level accuracy
   ├── Sensitivity
   ├── Specificity
   ├── Precision
   └── F1-score
   │
   ▼
Interpretability
   │
   ├── Saliency maps
   ├── PSD analysis
   ├── Frequency-band analysis
   ├── EEG topomaps
   └── t-SNE
```

---

# Features

* EEG preprocessing using **MNE**
* Band-pass filtering from **0.5–45 Hz**
* Downsampling to **125 Hz**
* Optional bipolar EEG conversion for the cross-validation preprocessing pipeline
* Fixed-length EEG segmentation
* 5-second windows with 4-second overlap
* Per-subject z-score normalization
* Subject-level cross-validation
* Pilot hyperparameter search
* Multiple deep-learning architectures
* GPU acceleration using PyTorch/CUDA
* Segment-level and subject-level evaluation
* Sensitivity and specificity calculation
* Precision and F1-score calculation
* Captum-based saliency analysis
* Power Spectral Density (PSD) analysis
* EEG frequency-band interpretation
* EEG scalp topographic visualization
* t-SNE feature visualization

---

# Models

The repository contains several EEG deep-learning architectures.

## 1. Oh_CNN

A 1D convolutional neural network operating directly on multichannel EEG signals.

Architecture:

```text
EEG
 │
 ▼
Conv1D
 │
 ▼
Max Pooling
 │
 ▼
Conv1D
 │
 ▼
Max Pooling
 │
 ▼
Conv1D
 │
 ▼
Average Pooling
 │
 ▼
Conv1D
 │
 ▼
Conv1D
 │
 ▼
Global Average Pooling
 │
 ▼
Fully Connected Layer
 │
 ▼
2 Classes
```

The implementation uses five convolutional stages with pooling and dropout.

---

## 2. SzHNN

A hybrid **CNN + LSTM** architecture.

```text
EEG
 │
 ▼
Conv1D
 │
 ▼
Max Pooling
 │
 ▼
Conv1D
 │
 ▼
Max Pooling
 │
 ▼
LSTM
 │
 ▼
Dense Layer
 │
 ▼
Dropout
 │
 ▼
2-Class Output
```

The CNN layers learn local temporal EEG patterns, while the LSTM processes the resulting temporal sequence.

---

## 3. EEGNet

An EEG-specific compact convolutional architecture.

It contains:

* Temporal convolution
* Depthwise spatial convolution
* Batch normalization
* ELU activation
* Average pooling
* Dropout
* Pointwise convolution
* Final classification layer

The implementation uses the number of EEG channels and number of temporal samples to dynamically determine the classifier input size.

---

## 4. SCCNet

A convolutional architecture designed around spatial and temporal EEG information.

The implementation applies:

```text
Spatial Convolution
        ↓
Batch Normalization
        ↓
Temporal Convolution
        ↓
Batch Normalization
        ↓
Squaring
        ↓
Average Pooling
        ↓
Log Transformation
        ↓
Fully Connected Classifier
```

---

## 5. ShallowConvNet

A shallow EEG convolutional network.

Its main processing stages are:

```text
Temporal Convolution
        ↓
Spatial Convolution
        ↓
Batch Normalization
        ↓
Square
        ↓
Average Pooling
        ↓
Log
        ↓
Dropout
        ↓
Linear Classifier
```

This architecture explicitly uses squared activations and logarithmic transformation, which are commonly useful operations for EEG signal representation.

---

## 6. MBSzEEGNet

A multi-branch EEG architecture combining multiple temporal scales.

The implementation contains:

* SCC-based branches
* EEGNet-based branches
* Multiple temporal kernel sizes
* Feature embeddings
* Combined branch representations
* Final classification head

The implementation uses temporal kernels corresponding to multiple time scales and combines their learned representations before classification.

---

## 7. EEG Conformer

The repository also contains an EEG Conformer implementation based on convolutional feature extraction followed by Transformer-style self-attention.

The architecture includes:

```text
EEG
 │
 ▼
Convolutional Patch Embedding
 │
 ▼
Projection
 │
 ▼
Multi-Head Self Attention
 │
 ▼
Residual Connections
 │
 ▼
Feed Forward Network
 │
 ▼
Transformer Encoder
```

The implementation uses `einops` components for tensor rearrangement and Transformer-style attention blocks.

---

# Dataset

The training pipeline supports two dataset configurations.

## Dataset 1

The default configuration contains:

```text
14 Control subjects
14 Schizophrenia subjects
```

for a total of:

```text
28 subjects
```

The code uses **19 EEG channels** for this dataset.

The channel configuration is:

```text
Fp2
F8
T4
T6
O2
Fp1
F7
T3
T5
O1
F4
C4
P4
F3
C3
P3
Fz
Cz
Pz
```

---

## Dataset 2

The second configuration contains:

```text
40 Control subjects
35 Schizophrenia subjects
```

for a total of:

```text
75 subjects
```

and uses 20 EEG channels during preprocessing.

---

# Dataset Directory Structure

The training code expects the preprocessed dataset to be organized approximately as:

```text
dataset 1/
│
├── Control/
│   ├── h01.set
│   ├── h02.set
│   ├── h03.set
│   └── ...
│
└── Schizophrenia/
    ├── s01.set
    ├── s02.set
    ├── s03.set
    └── ...
```

For Dataset 2:

```text
dataset 2/
│
├── Control/
│   ├── h01.npy
│   ├── h02.npy
│   └── ...
│
└── Schizophrenia/
    ├── s01.npy
    ├── s02.npy
    └── ...
```

The dataset loader determines the class from the folder and filename:

```text
Control       → class 0
Schizophrenia → class 1
```

---

# EEG Preprocessing

The preprocessing pipeline is implemented in `preprocess.py`.

## Band-pass filtering

EEG signals are filtered between:

```text
0.5 Hz – 45 Hz
```

This removes very-low-frequency drift and high-frequency components outside the selected range.

## Downsampling

The EEG recordings are resampled to:

```text
125 Hz
```

---

# Bipolar EEG Preprocessing

The `cross.py` script contains an additional preprocessing pipeline.

It converts EEG recordings into bipolar channels using electrode pairs such as:

```text
Fp1 - F3
F3  - C3
C3  - P3
P3  - O1

Fp2 - F4
F4  - C4
C4  - P4
P4  - O2

Fp1 - F7
F7  - T3
T3  - T5
T5  - O1

Fp2 - F8
F8  - T4
T4  - T6
T6  - O2
```

The bipolar signal is calculated as:

```text
bipolar_signal = channel_1 - channel_2
```

The resulting signals are then band-pass filtered and resampled to 125 Hz.

---

# EEG Segmentation

The training pipeline divides each recording into fixed-length windows.

Default parameters:

```text
Window length = 5 seconds
Overlap       = 4 seconds
Sampling rate = 125 Hz
```

Therefore, each EEG segment contains:

```text
5 × 125 = 625 samples
```

For example:

```text
Segment 1: 0s   → 5s
Segment 2: 1s   → 6s
Segment 3: 2s   → 7s
...
```

This segmentation is implemented using MNE's `make_fixed_length_epochs`.

---

# Normalization

After segmentation, each subject's EEG data is normalized using z-score normalization:

```text
x_normalized = (x - mean) / standard_deviation
```

The mean and standard deviation are calculated for that subject's segmented EEG data.

This is important because EEG recordings can have substantially different amplitude scales between subjects.

---

# Training Pipeline

The complete pipeline is controlled by:

```text
main.py
```

The default configuration is:

```text
Dataset              : 1
Sampling frequency   : 125 Hz
Window length        : 5 seconds
Window overlap       : 4 seconds
Pilot epochs         : 3
Full training epochs : 40
Cross-validation     : 5 folds
```

---

# Step 1 — Pilot Training

Before full training, the project performs a small hyperparameter search.

The grid contains:

### Batch size

```text
16
32
64
```

### Learning rate

```text
0.005
0.001
0.0005
0.0001
```

Therefore, each model evaluates:

```text
3 × 4 = 12
```

batch-size/learning-rate combinations.

The best combination is selected according to validation accuracy and saved in:

```text
checkpoints/pilot/pilot_result.txt
```

---

# Step 2 — Full Training

After selecting the best hyperparameters, each model is trained using the selected configuration.

The default number of training epochs is:

```text
40
```

For every fold:

```text
Training data
     ↓
Model training
     ↓
Validation
     ↓
Best validation model
     ↓
Save .pth checkpoint
```

The best model for each fold is stored under:

```text
checkpoints/full/<MODEL_NAME>/
```

For example:

```text
checkpoints/
└── full/
    └── EEGNet/
        ├── fold0_best_model.pth
        ├── fold1_best_model.pth
        ├── fold2_best_model.pth
        ├── fold3_best_model.pth
        └── fold4_best_model.pth
```

---

# Cross-Validation Strategy

The project uses **5-fold subject-level cross-validation**.

For Dataset 1:

```text
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

The subjects are divided into folds rather than randomly splitting individual EEG windows.

This is particularly important for EEG because multiple windows from the same subject are highly correlated. Keeping subjects separated between training and testing reduces the risk of evaluating on windows originating from subjects seen during training.

The code explicitly defines subject ranges for Dataset 1:

```text
[0–2]
[3–5]
[6–8]
[9–11]
[12–13]
```

---

# Loss Function and Optimizer

The models are trained using:

```text
Loss:
CrossEntropyLoss
```

and:

```text
Optimizer:
Adam
```

The learning rate is selected during pilot training.

---

# Testing

The testing stage evaluates the trained models at both:

### Segment level

Each 5-second EEG segment receives a prediction:

```text
0 → Control
1 → Schizophrenia
```

### Subject level

All segments belonging to a subject are evaluated, and the predicted subject class is determined using the most frequent predicted class across that subject's segments.

The code calculates:

```text
Segment Accuracy
Subject Accuracy
Sensitivity
Specificity
Precision
F1 Score
```

---

# Evaluation Metrics

## Accuracy

The percentage of correctly classified samples/subjects.

```text
Accuracy = Correct Predictions / Total Predictions
```

The project reports both:

```text
Segment Accuracy
Subject Accuracy
```

---

## Sensitivity

Sensitivity measures how many schizophrenia subjects are correctly identified.

```text
Sensitivity = TP / (TP + FN)
```

where:

```text
TP = Schizophrenia correctly classified
FN = Schizophrenia classified as Control
```

---

## Specificity

Specificity measures how many control subjects are correctly identified.

```text
Specificity = TN / (TN + FP)
```

where:

```text
TN = Control correctly classified
FP = Control classified as Schizophrenia
```

---

## Precision

```text
Precision = TP / (TP + FP)
```

---

## F1 Score

```text
F1 = 2 × Precision × Sensitivity
     ------------------------------
       Precision + Sensitivity
```

The testing implementation writes these metrics into:

```text
checkpoints/full/test_result.txt
```

---

# Explainable AI / Model Interpretation

The project does not stop at classification.

It also attempts to understand **which EEG signal components contribute to the model prediction**.

The interpretation pipeline uses **Captum Saliency**.

The saliency calculation estimates the contribution of individual EEG input values to the model's prediction.

---

# Power Spectral Density Analysis

The saliency signals are analyzed using Welch's Power Spectral Density estimation.

The implementation analyzes frequencies up to:

```text
40 Hz
```

The results are visualized across standard EEG frequency bands:

| Band  | Frequency |
| ----- | --------: |
| Delta |    0–4 Hz |
| Theta |    4–8 Hz |
| Alpha |   8–12 Hz |
| Beta  |  12–30 Hz |
| Gamma |  30–40 Hz |

The project generates PSD plots comparing the Control and Schizophrenia classes.

---

# EEG Topographic Maps

The project also generates scalp topographic maps.

These visualize the distribution of saliency/PSD-related information across EEG electrode locations.

Topomaps are generated for:

```text
All frequencies
Delta
Theta
Alpha
Beta
Gamma
```

The standard 10–20 EEG montage is used for electrode positions.

---

# t-SNE Visualization

The testing pipeline also extracts learned representations from the neural networks and uses **t-SNE** to project them into two dimensions.

Conceptually:

```text
EEG
 ↓
Neural Network
 ↓
Learned Feature Representation
 ↓
t-SNE
 ↓
2D Visualization
```

This allows visual inspection of whether learned representations form distinguishable groups corresponding to:

```text
Control
vs
Schizophrenia
```

The implementation uses scikit-learn's `TSNE` and `MinMaxScaler`.

---

# Project Structure

```text
Schizophernia_detection_using-brain_signal/
│
├── main.py
│
├── preprocess.py
├── cross.py
│
├── func.py
│
├── models.py
│
├── pilot_train.py
├── full_train.py
├── test.py
│
├── interpretation.py
│
├── check.py
├── asr.py
├── kmuh_get_rsEEG.py
│
├── main2.py
├── test2.py
│
└── requirements.txt
```

---

# File Description

| File                | Purpose                                                                             |
| ------------------- | ----------------------------------------------------------------------------------- |
| `main.py`           | Main pipeline controller                                                            |
| `preprocess.py`     | EEG channel processing, filtering and downsampling                                  |
| `cross.py`          | Bipolar EEG preprocessing pipeline                                                  |
| `func.py`           | Dataset loading, normalization, tensor conversion and training/evaluation utilities |
| `models.py`         | Deep-learning model architectures                                                   |
| `pilot_train.py`    | Hyperparameter search                                                               |
| `full_train.py`     | Full model training and checkpoint generation                                       |
| `test.py`           | Model testing and metric calculation                                                |
| `interpretation.py` | Saliency, PSD and topographic interpretation                                        |
| `asr.py`            | Additional EEG preprocessing functionality                                          |
| `kmuh_get_rsEEG.py` | Dataset-specific EEG processing                                                     |
| `check.py`          | Dataset/checking utility                                                            |
| `main2.py`          | Additional experiment entry point                                                   |
| `test2.py`          | Additional testing functionality                                                    |
| `requirements.txt`  | Python dependency specification                                                     |

The repository currently contains these source files and has one commit on its GitHub `main` branch.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/manish123-ui/Schizophernia_detection_using-brain_signal.git

cd Schizophernia_detection_using-brain_signal
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The repository specifies NumPy, Matplotlib, SciPy, PyTorch, Captum and EDFlib-Python as dependencies.

---

# Requirements

The current `requirements.txt` contains:

```text
numpy>=2.1.0
matplotlib>=3.9.0
scipy>=1.14.0
torch>=2.5.0
captum>=0.7.0
EDFlib-Python>=1.0.8
```

The code also imports libraries such as:

```text
MNE
scikit-learn
einops
torchvision
PIL
```

so these packages may need to be installed depending on the execution path used.

---

# Preprocessing

For Dataset 1, if the raw EEG data is available in the expected directory structure, preprocessing can be performed with:

```bash
python preprocess.py --dataset 1
```

The preprocessing script applies:

```text
0.5–45 Hz band-pass filtering
↓
125 Hz resampling
```

For Dataset 2, the script additionally performs channel selection before filtering and resampling.

---

# Training

The complete pipeline can be launched using:

```bash
python main.py
```

The default configuration is equivalent to:

```bash
python main.py \
    --dataset 1 \
    --dataset_path "./dataset 1/" \
    --savepath "./checkpoints/" \
    --pilot_train_epoch 3 \
    --full_train_epoch 40 \
    --fold 5 \
    --model All \
    --sfreq 125 \
    --window_length 5 \
    --window_overlap 4
```

The program sequentially performs:

```text
Pilot Training
      ↓
Hyperparameter Selection
      ↓
Full Training
      ↓
Testing
      ↓
t-SNE
      ↓
Saliency Analysis
      ↓
PSD Analysis
      ↓
EEG Topographic Maps
```

This execution flow is defined directly in `main.py`.

---

# Training a Specific Model

Instead of training all models, a specific architecture can be selected.

For example:

```bash
python main.py --model EEGNet
```

Available model names in the main pipeline include:

```text
Oh_CNN
SzHNN
EEGNet
SCCNet
ShallowConvNet
MBSzEEGNet
Conformer
```

The default:

```bash
--model All
```

runs the supported model list sequentially.

---

# Custom Parameters

Example:

```bash
python main.py \
    --dataset 1 \
    --model EEGNet \
    --sfreq 125 \
    --window_length 5 \
    --window_overlap 4 \
    --pilot_train_epoch 3 \
    --full_train_epoch 40 \
    --fold 5
```

Available arguments include:

```text
--dataset
--dataset_path
--savepath
--pilot_train_epoch
--full_train_epoch
--fold
--model
--sfreq
--channel
--window_length
--window_overlap
--seed
```

---

# Output Directory

After training, the expected output structure is approximately:

```text
checkpoints/
│
├── pilot/
│   └── pilot_result.txt
│
├── full/
│   ├── EEGNet/
│   │   ├── fold0_best_model.pth
│   │   ├── fold1_best_model.pth
│   │   ├── fold2_best_model.pth
│   │   ├── fold3_best_model.pth
│   │   ├── fold4_best_model.pth
│   │   └── ...
│   │
│   └── test_result.txt
│
├── xai/
│   ├── EEGNet_psd.png
│   └── ...
│
└── tsne/
    └── ...
```

The exact generated files depend on the selected model and execution path.

---

# Prediction Classes

The model performs binary classification:

```text
Class 0 → Control / Healthy Control

Class 1 → Schizophrenia
```

The class labels are assigned in the dataset loader.

Therefore, at a high level, the model answers:

```text
EEG Signal
     │
     ▼
Deep Learning Model
     │
     ├──────────────┐
     ▼              ▼
Class 0           Class 1
Control           Schizophrenia
```

---

# Important: Segment vs Subject Prediction

A key design choice in this project is that a single EEG recording is divided into many overlapping segments.

For example:

```text
One subject
    │
    ├── Segment 1 → prediction
    ├── Segment 2 → prediction
    ├── Segment 3 → prediction
    ├── ...
    └── Segment N → prediction
```

The final subject-level prediction is based on the most frequent predicted class among that subject's segments.

This means that a subject can have:

```text
Segment predictions:

Control
Control
Schizophrenia
Control
Control
...
```

and the final subject prediction can still be:

```text
Control
```

because it is the majority prediction.

---

# Reproducibility

The project provides deterministic seed configuration through Python, NumPy and PyTorch.

The implementation sets:

```text
random seed
NumPy seed
PyTorch seed
CUDA seed
```

and configures cuDNN for deterministic behavior.

Dataset-specific default seeds are also defined in `main.py`.

---

# Model Interpretation

A major component of the project is interpretability.

Instead of only producing:

```text
Control / Schizophrenia
```

the framework also analyzes which EEG regions and frequency components are associated with model predictions.

The interpretation pipeline includes:

### Saliency

Identifies input regions that contribute to predictions.

### PSD

Analyzes frequency-domain characteristics of the saliency signals.

### Frequency bands

```text
Delta
Theta
Alpha
Beta
Gamma
```

### Topographic visualization

Maps EEG information onto scalp locations.

### t-SNE

Visualizes learned feature representations in 2D.

---

# Research Workflow

The overall experimental workflow can be summarized as:

```text
             RAW EEG
                │
                ▼
       ┌─────────────────┐
       │ EEG Preprocess  │
       └─────────────────┘
                │
                ▼
       0.5–45 Hz Filter
                │
                ▼
        Resample 125 Hz
                │
                ▼
       EEG Windowing
       5 sec / 4 sec overlap
                │
                ▼
       Subject Normalization
                │
                ▼
       ┌─────────────────┐
       │ Pilot Training  │
       └─────────────────┘
                │
                ▼
    Batch Size + Learning Rate
           Selection
                │
                ▼
       ┌─────────────────┐
       │ Full Training   │
       │   5-Fold CV     │
       └─────────────────┘
                │
                ▼
       Best Fold Models
                │
                ▼
            Testing
                │
        ┌───────┴────────┐
        ▼                ▼
 Segment Accuracy   Subject Accuracy
        │                │
        └───────┬────────┘
                ▼
       Sensitivity / Specificity
       Precision / F1 Score
                │
                ▼
        Explainability
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
    Saliency    PSD      t-SNE
       │
       ▼
 EEG Topographic Maps
```

---

# Results

The repository's code is designed to generate quantitative results rather than hard-coding a single accuracy value.

Testing writes:

```text
Accuracy (Seg.)
Accuracy (Sub.)
Sensitivity
Specificity
Precision
F1_score
```

to:

```text
checkpoints/full/test_result.txt
```

Therefore, the actual results should be reported from the generated `test_result.txt` produced by your experiment rather than claiming a performance number that is not stored in the repository.

---

# Limitations

This project has several important limitations.

### 1. Research classification ≠ clinical diagnosis

The model distinguishes EEG recordings from the two classes represented in the dataset. This does not establish that the system can independently diagnose schizophrenia in clinical practice.

### 2. Small subject counts

Dataset 1 contains only:

```text
14 Control
14 Schizophrenia
```

subjects in the configured pipeline.

### 3. EEG dataset dependence

Performance can depend heavily on:

* EEG acquisition equipment
* Electrode configuration
* Sampling rate
* Reference scheme
* Recording protocol
* Subject population
* Medication status
* Artifact characteristics

### 4. Segment correlation

The pipeline creates heavily overlapping 5-second windows with 4 seconds of overlap. Therefore, adjacent segments are highly correlated.

Subject-level splitting is used in the training pipeline, which helps avoid directly placing the same subject in training and validation/test sets, but the strong correlation among windows remains an important consideration when interpreting segment-level metrics.

### 5. External validation

Strong performance on one EEG dataset does not automatically imply equivalent performance on another population or recording system.

External validation on an independent dataset is required before making claims about generalization.

---

# Technologies Used

### Programming Language

```text
Python
```

### Deep Learning

```text
PyTorch
```

### EEG Processing

```text
MNE-Python
```

### Explainable AI

```text
Captum
```

### Scientific Computing

```text
NumPy
SciPy
```

### Visualization

```text
Matplotlib
```

### Feature Visualization

```text
scikit-learn
t-SNE
```

### Tensor Manipulation

```text
einops
```

---

# Citation / Background

EEG-based machine-learning approaches for schizophrenia classification have been investigated in the research literature. For example, published work has explored deep residual networks and other deep-learning approaches for identifying schizophrenia-related patterns from EEG signals.

Other research has investigated feature selection and multichannel EEG representations for schizophrenia detection.

This repository follows the broader research direction of using EEG signal processing and deep learning for binary classification of schizophrenia and control groups.

---

# Disclaimer

This software is provided for **research and educational purposes only**.

The output of the machine-learning model should **not** be used as a standalone medical diagnosis or as a substitute for evaluation by a qualified healthcare professional.

The model learns statistical patterns from the datasets used for training and evaluation. Its predictions may not generalize to individuals, hospitals, EEG devices, acquisition protocols, or populations that differ from the training data.

---

# Author

**Manish Kumar**

GitHub:

https://github.com/manish123-ui

Project:

https://github.com/manish123-ui/Schizophernia_detection_using-brain_signal

---

# License

No explicit license is currently provided in the repository.

If you intend to distribute or reuse this project, add an appropriate open-source license to the repository.
