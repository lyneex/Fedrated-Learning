"""
src/config.py -- Central configuration for the Federated Cardio project.
Edit this file to change paths, hyperparameters, and experiment settings.
"""
import os

# -- Paths ---------------------------------------------------------------------
# Project root = parent of the src/ directory
PROJECT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(PROJECT_DIR, "cardio_train.csv")
RUNS_DIR      = os.path.join(PROJECT_DIR, "runs")
CHECKPOINT_DIR = os.path.join(RUNS_DIR, "checkpoints")
BASELINE_DIR  = os.path.join(RUNS_DIR, "baseline")
FEDERATED_DIR = os.path.join(RUNS_DIR, "federated")

# -- Reproducibility -----------------------------------------------------------
SEED = 42

# -- Data Splitting ------------------------------------------------------------
TEST_SIZE   = 0.20       # fraction held out as the shared test set
NUM_CLIENTS = 3          # number of simulated hospitals
NON_IID     = True       # True -> non-IID hospital split; False -> IID random split

# -- Model Architecture --------------------------------------------------------
HIDDEN_DIMS = [64, 32]   # neurons in each hidden layer
DROPOUT     = 0.30       # dropout probability

# -- Multimodal Settings -------------------------------------------------------
MULTIMODAL       = True       # True -> use multimodal fusion; False -> legacy MLP
EMBED_DIM        = 256        # per-encoder embedding dimension
FUSED_DIM        = 512        # fused patient representation dimension
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
BATCH_SIZE       = 64
BASELINE_EPOCHS  = 20    # epochs for the centralized baseline
LOCAL_EPOCHS     = 3     # local epochs per federated round
NUM_ROUNDS       = 10    # total federated communication rounds
