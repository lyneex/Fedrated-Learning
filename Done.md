# Federated Cardio — Implementation Inventory

## Project outcome

This project implements binary cardiovascular-disease prediction using a
PyTorch multilayer perceptron (MLP). It supports two comparable training
approaches:

1. **Centralized baseline training:** one model is trained on the combined
   data from all simulated hospitals.
2. **Federated learning:** three simulated hospitals train private local model
   copies and a server aggregates their parameters with sample-weighted
   Federated Averaging (FedAvg). Raw records remain within each hospital
   partition during this process.

The repository includes the input data, preprocessing pipeline, model,
optional Flower client/server implementation, executable baseline and manual
FedAvg workflows, saved models/checkpoints, metrics, and visual reports.

## Implemented workflow

```
cardio_train.csv
  → validation/cleaning/feature engineering
  → stratified 80/20 train/test split and training-only standardization
  → three simulated, non-IID hospital datasets
  → centralized MLP baseline OR local MLP training + weighted FedAvg
  → test and per-hospital evaluation
  → JSON metrics, PyTorch weights/checkpoints, and PNG charts
```

The default experiment uses seed 42, a 12-feature MLP with hidden layers
`64 → 32`, dropout 0.30, Adam with learning rate 0.001, 64-sample batches,
20 baseline epochs, and 10 federated rounds with three local epochs each.

## Source and configuration files

| File | What has been implemented |
| --- | --- |
| `README.md` | Brief project introduction defining federated learning as collaborative model training without sharing raw data. |
| `requirements.txt` | Declares the required stack: PyTorch, NumPy, pandas, scikit-learn, matplotlib, seaborn, Flower, and tqdm. |
| `baseline.py` | Runnable centralized-training experiment. It combines the three hospital training datasets, trains the MLP with binary cross-entropy-with-logits loss, saves model weights and test metrics, and creates training-history and confusion-matrix charts. |
| `federated_main.py` | Runnable pure-Python federated simulation. It broadcasts global weights, trains one cloned model per hospital, computes sample-weighted FedAvg, evaluates each round, saves all ten global checkpoints, then writes final/per-hospital metrics and visual comparisons. It does not require Flower/Ray for this execution path. |
| `src/__init__.py` | Marks `src` as the Federated Cardio Python package. |
| `src/config.py` | Central experiment settings: paths, seed, shared-test fraction, number of clients, non-IID switch, model dimensions/dropout, and training hyperparameters. **Note:** `DATA_PATH` is currently a machine-specific Windows path; to run the scripts from this checkout it should be changed to the local `cardio_train.csv` path. |
| `src/data_utils.py` | Complete data pipeline: reads the semicolon-delimited CSV; drops IDs/duplicates and invalid physiological values; derives age in years, BMI, pulse pressure, and hypertension; recodes gender; stratifies the train/test split; fits a `StandardScaler` only on training numeric fields; creates PyTorch `TensorDataset`s; and performs either IID partitions or the configured three-way non-IID hospital split. It also saves the fitted scaler. |
| `src/model.py` | Defines `CardioMLP`, a binary classifier with Linear → LayerNorm → ReLU → Dropout blocks and a one-logit output. Also supplies NumPy parameter import/export helpers for federated exchange and PyTorch checkpoint save/load helpers. |
| `src/client.py` | Flower `NumPyClient` implementation for a simulated hospital. It receives global parameters, trains only on its private tensor dataset, returns updated weights/sample count/local loss, and evaluates received global weights locally. |
| `src/server.py` | Flower FedAvg strategy builder. It seeds global parameters, configures client participation and per-round hyperparameters, evaluates the aggregated model on the shared test set, records round metrics, and saves round checkpoints. |
| `src/evaluate.py` | Reusable binary-classification evaluation: BCE loss, accuracy, precision, recall, F1, ROC-AUC, and a 2×2 confusion matrix. It also prints reports and writes metrics JSON. |
| `src/visualize.py` | Non-interactive plotting helpers for centralized loss/accuracy history, federated accuracy and loss curves, confusion-matrix heatmaps, and grouped centralized/local/federated metric comparisons. |

## Dataset and persisted preprocessing artifact

| File | What has been implemented / stored |
| --- | --- |
| `cardio_train.csv` | The 70,000-row, semicolon-delimited cardiovascular dataset used by the project. It contains demographic, lifestyle, clinical, and target (`cardio`) fields; `id` is removed during cleaning. |
| `runs/scaler.pkl` | Serialized training-fitted `StandardScaler` for the seven numeric model features. It enables future inference to use exactly the training feature scale. |

## Centralized baseline outputs

| File | What it contains |
| --- | --- |
| `runs/baseline/baseline_model.pt` | Saved PyTorch state dictionary for the centralized MLP trained on the combined hospital training data. |
| `runs/baseline/metrics.json` | Centralized test results: loss **0.5458**, accuracy **72.96%**, precision **75.20%**, recall **67.66%**, F1 **0.7123**, ROC-AUC **0.7983**, plus the confusion matrix. |
| `runs/baseline/training_history.png` | Two-panel plot of the baseline model's training BCE loss and accuracy across its 20 epochs. |
| `runs/baseline/confusion_matrix.png` | Heatmap of centralized test predictions against actual classes. |

## Federated-training outputs

| File | What it contains |
| --- | --- |
| `runs/checkpoints/round_01.pt` | Global PyTorch model state after federated round 1. |
| `runs/checkpoints/round_02.pt` | Global PyTorch model state after federated round 2. |
| `runs/checkpoints/round_03.pt` | Global PyTorch model state after federated round 3. |
| `runs/checkpoints/round_04.pt` | Global PyTorch model state after federated round 4. |
| `runs/checkpoints/round_05.pt` | Global PyTorch model state after federated round 5. |
| `runs/checkpoints/round_06.pt` | Global PyTorch model state after federated round 6. |
| `runs/checkpoints/round_07.pt` | Global PyTorch model state after federated round 7. |
| `runs/checkpoints/round_08.pt` | Global PyTorch model state after federated round 8. |
| `runs/checkpoints/round_09.pt` | Global PyTorch model state after federated round 9. |
| `runs/checkpoints/round_10.pt` | Final global PyTorch model state after federated round 10. This checkpoint is reloaded for final evaluation. |
| `runs/federated/rounds_log.json` | Per-round global loss, accuracy, precision, recall, F1, and ROC-AUC for all 10 FedAvg rounds. |
| `runs/federated/final_metrics.json` | Final global-model test results: loss **0.5484**, accuracy **72.70%**, precision **76.31%**, recall **65.00%**, F1 **0.7020**, ROC-AUC **0.7977**, plus the confusion matrix. |
| `runs/federated/local_hospital_metrics.json` | Evaluation of the final global model on each non-IID hospital dataset: H0 (older), H1 (hypertensive), and H2 (balanced). |
| `runs/federated/accuracy_curve.png` | Global test accuracy over the 10 federated rounds. |
| `runs/federated/loss_curve.png` | Global BCE test loss over the 10 federated rounds. |
| `runs/federated/confusion_matrix.png` | Heatmap of final federated global-model test predictions. |
| `runs/federated/model_comparison.png` | Grouped comparison of accuracy, F1, and ROC-AUC for the centralized baseline, each hospital evaluation, and the final federated global model. |

## Generated Python cache files

The following files are Python bytecode caches generated automatically during
module imports. They are not handwritten source and can be safely regenerated:

- `src/__pycache__/__init__.cpython-313.pyc`
- `src/__pycache__/client.cpython-313.pyc`
- `src/__pycache__/config.cpython-313.pyc`
- `src/__pycache__/data_utils.cpython-313.pyc`
- `src/__pycache__/evaluate.cpython-313.pyc`
- `src/__pycache__/model.cpython-313.pyc`
- `src/__pycache__/server.cpython-313.pyc`
- `src/__pycache__/visualize.cpython-313.pyc`

## Current comparison summary

The saved run shows that federated training reached **72.70%** test accuracy
and **0.7977** ROC-AUC, close to the centralized baseline at **72.96%** and
**0.7983** respectively. This demonstrates the project's central goal: a
global model can be trained through parameter aggregation while hospital raw
training records remain partitioned.
