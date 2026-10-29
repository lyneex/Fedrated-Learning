"""
src/config.py -- Central configuration for the Federated Cardio project.
Edit this file to change paths, hyperparameters, and experiment settings.
"""
import os

# -- Paths ---------------------------------------------------------------------
# Project root = parent of the src/ directory
PROJECT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(PROJECT_DIR, "data")

# -- Input mode ---------------------------------------------------------------
# "tabular" uses cardio_train.csv and CardioMLP.
# "cardiovascular_tabular" uses the Kaggle clinical CSV and CardioMLP.
# "image" uses IMAGE_MANIFEST_PATH and ImageClassifier.
# "ecg_image" uses the Kaggle ECG class folders and ImageClassifier.
# "multimodal" uses the real linked MIMIC pipeline in src/mimic_data.py.
DATA_MODE = os.environ.get("DATA_MODE", "cardiovascular_tabular")

DATA_PATH     = os.path.join(DATA_DIR, "tabular", "cardio_train.csv")
RUNS_DIR      = os.path.join(PROJECT_DIR, "runs")
EXPERIMENT_DIR = os.path.join(RUNS_DIR, DATA_MODE)
CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
BASELINE_DIR  = os.path.join(EXPERIMENT_DIR, "baseline")
FEDERATED_DIR = os.path.join(EXPERIMENT_DIR, "federated")

# -- Reproducibility -----------------------------------------------------------
SEED = 42

# -- Data Splitting ------------------------------------------------------------
TEST_SIZE   = 0.20       # fraction held out as the shared test set
VALIDATION_SIZE = 0.10   # fraction of the remaining training data for model selection
NUM_CLIENTS = 3          # number of simulated hospitals
NON_IID     = True       # True -> non-IID hospital split; False -> IID random split

# -- Model Architecture --------------------------------------------------------
HIDDEN_DIMS = [64, 32]   # neurons in each hidden layer
DROPOUT     = 0.30       # dropout probability

# -- Real image dataset settings ---------------------------------------------
# Create data/image_manifest.csv with at least these columns:
#   image_path,label
# image_path is relative to IMAGE_ROOT (or an absolute path); label is 0 or 1.
# MIMIC-CXR-JPG's CheXpert/NegBio CSV can be converted into this small manifest
# after access has been approved through PhysioNet.
IMAGE_ROOT          = os.path.join(DATA_DIR, "images")
IMAGE_MANIFEST_PATH = os.path.join(DATA_DIR, "manifests", "image_manifest.csv")
IMAGE_PATH_COLUMN   = "image_path"
IMAGE_LABEL_COLUMN  = "label"

# -- MIMIC-IV + MIMIC-CXR real multimodal settings ---------------------------
# These paths deliberately point to local, credentialed downloads. Never commit
# MIMIC data or send it to services that are not permitted by its DUA.
MIMIC_DB_DIR            = os.path.join(DATA_DIR, "mimic-cxr-db")
MIMIC_ADMISSIONS_PATH   = os.path.join(MIMIC_DB_DIR, "admissions.csv")
MIMIC_RADIOGRAPHS_PATH  = os.path.join(MIMIC_DB_DIR, "radiographs.csv")
MIMIC_CHARTEVENTS_PATH  = os.path.join(DATA_DIR, "mimic-iv", "icu", "chartevents.csv.gz")
MIMIC_CXR_ROOT          = os.path.join(IMAGE_ROOT, "mimic-cxr-jpg")
MIMIC_COHORT_PATH       = os.path.join(DATA_DIR, "manifests", "mimic_multimodal.csv")
MIMIC_VITALS_CACHE_PATH = os.path.join(DATA_DIR, "manifests", "mimic_vitals.npz")
MIMIC_TARGET_COLUMN     = "Cardiomegaly"  # binary radiograph label from radiographs.csv
MIMIC_MAX_SAMPLES       = 5_000            # safe starter cohort; None uses every eligible row
MIMIC_CHARTEVENTS_CHUNK_SIZE = 1_000_000

# Canonical MIMIC-IV item IDs for [heart rate, SpO2, systolic BP, diastolic BP].
# Both CareVue and MetaVision IDs are accepted where applicable.
MIMIC_VITAL_ITEM_IDS = {
    "heart_rate": (211, 220045),
    "spo2": (646, 220277),
    "sbp": (51, 220050, 220179),
    "dbp": (836, 220051, 220180),
}

# -- Student-scale Kaggle cardiovascular dataset -----------------------------
# Its clinical rows and ECG images have no verified patient/image mapping, so
# use the two modes below as separate experiments, never as fused multimodality.
KAGGLE_CARDIO_DIR = os.path.join(DATA_DIR, "cardiovascular_multimodal")
KAGGLE_CARDIO_CSV = os.path.join(KAGGLE_CARDIO_DIR, "cardiovascular_dataset.csv")
KAGGLE_ECG_IMAGE_ROOT = os.path.join(IMAGE_ROOT, "ecg_images")

# -- Multimodal Settings -------------------------------------------------------
# Kept for backwards compatibility. DATA_MODE determines the model to run.
MULTIMODAL       = False
EMBED_DIM        = 128        # reduced for 4 GB-class GPUs; per-encoder embedding dimension
FUSED_DIM        = 256        # reduced for 4 GB-class GPUs; fused patient representation
VITALS_SEQ_LEN   = 64         # time steps in vitals sequences
VITALS_CHANNELS  = 4          # number of vital sign channels (HR, SpO2, SBP, DBP)
IMG_SIZE         = 64         # imaging spatial resolution (IMG_SIZE × IMG_SIZE)
IMG_CHANNELS     = 1          # grayscale

# Per-hospital modality availability (indices → hospital IDs)
# Each entry: (has_ehr, has_vitals, has_imaging)
HOSPITAL_MODALITIES = {
    0: (True, True, True),     # H0: all modalities
    1: (True, True, False),    # H1: tabular + vitals (no imaging)
    2: (True, False, False),   # H2: tabular only
}

# -- Training ------------------------------------------------------------------
LEARNING_RATE    = 1e-3
BATCH_SIZE       = 16    # conservative starting point for a GTX 1650 Ti-class GPU
BASELINE_EPOCHS  = 30    # epochs for the centralized baseline
LOCAL_EPOCHS     = 3     # local epochs per federated round
NUM_ROUNDS       = 20    # total federated communication rounds
EARLY_STOPPING_PATIENCE = 5
USE_POS_WEIGHT   = True  # weight the minority class using training labels only
THRESHOLD_START  = 0.05
THRESHOLD_STOP   = 0.95
THRESHOLD_STEP   = 0.01
