# Multimodal Federated Learning for Cardiovascular Disease Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flower FL](https://img.shields.io/badge/Flower-FL-ffd43b.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A decentralized, privacy-preserving deep learning framework that trains cardiovascular disease prediction models across distributed hospital clients without sharing patient data. It runs either tabular cardiovascular records or real labeled medical images; a real, patient-linked multimodal dataset is required before claiming a multimodal experiment.

---

## 🌟 Key Features

1. **Multimodal Fusion Architecture**:
   - **Tabular EHR Encoder**: Multilayer perceptron with LayerNorm, LeakyReLU, and Dropout.
   - **Vitals Time-Series Encoder**: 1D Convolutional Neural Network + Bidirectional GRU for continuous vital signs (Heart Rate, $\text{SpO}_2$, SBP, DBP).
   - **Imaging Encoder**: Custom CNN / ResNet-style feature extractor for medical imaging (e.g. chest X-ray / echocardiogram representations).
   - **Gated Cross-Attention Fusion**: Dynamically weighs and integrates modality embeddings into a shared patient representation.

2. **Graceful Missing-Modality Handling**:
   - Hospitals often have heterogeneous data collection capabilities (e.g., Hospital 0 has all modalities, Hospital 1 lacks imaging, Hospital 2 only has EHR).
   - Missing data streams are zero-filled with an explicit **presence mask** $\mathbf{m} \in \{0, 1\}^3$ so models explicitly learn modality availability rather than hallucinating features.

3. **Sample-Weighted Federated Averaging (FedAvg)**:
   - Private client updates are aggregated using FedAvg weighted by local sample sizes.
   - Preserves patient confidentiality—only model weights are communicated.

4. **Realistic Non-IID Hospital Partitioning**:
   - Simulates demographic and clinical heterogeneity across hospital populations.

---

## 📊 Evaluation & Test Results

Evaluated on held-out test data across 10 communication rounds ($K=3$ clients, non-IID partition):

| Metric | Centralized Baseline | Federated Multimodal (FedAvg) |
| :--- | :---: | :---: |
| **ROC-AUC** | **0.7983** | **0.7977** |
| **Accuracy** | **72.96%** | **72.70%** |
| **Precision** | 75.20% | **76.31%** |
| **Recall** | **67.66%** | 65.00% |
| **F1 Score** | **0.7123** | 0.7020% |
| **Test Loss** | **0.5458** | 0.5484 |

> **Key Takeaway**: Federated learning matches the centralized benchmark within **0.06% ROC-AUC** and **0.26% accuracy** while strictly preserving data privacy and handling missing imaging/vitals across clients.

### Visual Test Results

- **Model Comparison**: `runs/federated/model_comparison.png`
- **Federated Convergence (Accuracy)**: `runs/federated/accuracy_curve.png`
- **Federated Convergence (Loss)**: `runs/federated/loss_curve.png`
- **Confusion Matrix**: `runs/federated/confusion_matrix.png`
- **Baseline Training History**: `runs/baseline/training_history.png`

---

## 📂 Repository Structure

```
├── data/
│   ├── tabular/cardio_train.csv # Kaggle cardiovascular disease dataset (70,000 records)
│   ├── images/                  # Local real images; ignored by Git
│   └── manifests/               # Image-path and label CSV files
├── baseline.py                  # Centralized baseline training script
├── federated_main.py            # Federated simulation runner (FedAvg across hospitals)
├── requirements.txt             # Project dependencies
├── Methodology_Project.md       # Detailed research & architectural methodology
├── Done.md                      # Implementation inventory and verification log
├── multimodal_integration_plan.md# Multimodal integration specification
├── src/
│   ├── __init__.py
│   ├── config.py                # Hyperparameters, paths, and modality availability
│   ├── data_utils.py            # Data preprocessing, splitting, and multimodal pipelines
│   ├── synthetic_data.py        # Coherent synthetic vitals & imaging generators
│   ├── model.py                 # Model factory & legacy CardioMLP architecture
│   ├── multimodal_model.py      # Modality encoders & gated cross-attention fusion
│   ├── client.py                # Flower-compatible hospital client implementation
│   ├── server.py                # Flower aggregation strategy & checkpoints
│   ├── evaluate.py              # Metrics computation (AUC, F1, Accuracy, etc.)
│   └── visualize.py             # Plotting utilities for training curves & comparisons
└── runs/                        # Test results, checkpoints, and generated plots
    ├── baseline/                # Baseline metrics and charts
    ├── federated/               # Federated round logs, metrics, and comparisons
    ├── checkpoints/             # Per-round global model checkpoints (.pt)
    └── scaler.pkl               # Fitted standard scaler
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/lyneex/Fedrated-Learning.git
cd Fedrated-Learning
pip install -r requirements.txt
```

### 2. Run tabular cardiovascular training
The checked-in `cardio_train.csv` contains 70,000 tabular patient records; it has no images.
```bash
python3 baseline.py
```

### 3. Run Federated Simulation
Execute 10 rounds of federated training across 3 simulated hospitals with heterogeneous modality availability:
```bash
python3 federated_main.py
```

Outputs are isolated by input mode, e.g. `runs/tabular/`, `runs/image/`, and
`runs/multimodal/`, so models from different experiments are never compared by
accident.

### 4. Run image recognition

After downloading an approved image dataset, set `DATA_MODE = "image"` in
`src/config.py`. Copy `data/manifests/image_manifest.example.csv` to
`data/manifests/image_manifest.csv` and replace its example rows with two columns:

```csv
image_path,label
relative/path/to/image.jpg,0
relative/path/to/positive-image.jpg,1
```

Paths are relative to `data/images/` by default. The image CNN reads images
lazily, converts them to grayscale, resizes them to 64×64, and trains with the
same baseline or federated commands above. Labels must be binary (`0` or `1`)
and both classes need at least two samples.

### 5. Run real EHR + vitals + chest-X-ray training (MIMIC)

After receiving PhysioNet approval, place the MIMIC-CXR-DB `admissions.csv` and
`radiographs.csv` in `data/mimic-cxr-db/`, MIMIC-IV `chartevents.csv.gz` in
`data/mimic-iv/icu/`, and downloaded MIMIC-CXR-JPG images below
`data/images/mimic-cxr-jpg/files/`. Then set `DATA_MODE = "multimodal"` and run:

```bash
python3 prepare_mimic_data.py
python3 baseline.py
python3 federated_main.py
```

The preparation script retains only admission-linked radiographs with a
definite binary target, joins ICU vital chart events by `hadm_id`, resamples the
four vital streams, and prevents a patient from appearing in both train and
test sets. MIMIC files are ignored by Git.

### 6. Run the small Kaggle cardiovascular experiments

The Kaggle release included here has a clinical CSV and ECG image folders, but
no verified mapping from a CSV row to an ECG image. They are intentionally run
as separate binary experiments, where `Normal=0` and every other diagnosis is
`1`; they are not patient-level multimodal fusion.

```python
# src/config.py
DATA_MODE = "cardiovascular_tabular"  # clinical data, 4,327 records
# or
DATA_MODE = "ecg_image"               # ECG-image CNN, 928 images
```

For either mode, run the centralized baseline first and then the federated
simulation:

```bash
python3 baseline.py
python3 federated_main.py
```

To run both separate experiments with one command:

```bash
python3 run_kaggle_experiments.py
```
