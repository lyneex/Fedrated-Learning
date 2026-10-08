"""
src/synthetic_data.py -- Synthetic multimodal data generators.

Generates simulated time-series vitals and medical imaging from existing
tabular features.  Used for prototyping the multimodal pipeline when only
tabular EHR data is available.

In production, these functions would be replaced by real data ingestion
pipelines for each modality.
"""
import numpy as np
import src.config as config


def generate_vitals(X_tabular: np.ndarray, seed: int = config.SEED) -> np.ndarray:
    """
    Generate synthetic vitals time-series from tabular features.

    Uses ap_hi (col 3), ap_lo (col 4) to seed blood-pressure channels,
    and derives heart-rate / SpO2 from BMI (col 5) and age (col 0).

    Parameters
    ----------
    X_tabular : (N, n_features) -- standardised tabular feature matrix
    seed      : random seed for reproducibility

    Returns
    -------
    vitals : (N, T, C) float32 array
             N patients, T=VITALS_SEQ_LEN timesteps, C=VITALS_CHANNELS channels
    """
    rng = np.random.default_rng(seed)
    N = X_tabular.shape[0]
    T = config.VITALS_SEQ_LEN
    C = config.VITALS_CHANNELS

    vitals = np.zeros((N, T, C), dtype=np.float32)

    for i in range(N):
        # Channel 0: Simulated heart rate (influenced by age_years = col 0)
        base_hr = 70.0 + X_tabular[i, 0] * 5.0
        vitals[i, :, 0] = base_hr + rng.normal(0, 3.0, T)

        # Channel 1: Simulated SpO2 (influenced by BMI = col 5)
        base_spo2 = 97.0 - abs(X_tabular[i, 5]) * 0.5
        vitals[i, :, 1] = base_spo2 + rng.normal(0, 0.5, T)

        # Channel 2: Systolic BP (influenced by ap_hi = col 3)
        base_sbp = 120.0 + X_tabular[i, 3] * 10.0
        trend = np.linspace(0, rng.uniform(-5, 5), T)
        vitals[i, :, 2] = base_sbp + trend + rng.normal(0, 4.0, T)

        # Channel 3: Diastolic BP (influenced by ap_lo = col 4)
        base_dbp = 80.0 + X_tabular[i, 4] * 5.0
        vitals[i, :, 3] = base_dbp + trend * 0.6 + rng.normal(0, 3.0, T)

    # Z-score normalise per channel across all patients
    for c in range(C):
        mu = vitals[:, :, c].mean()
        sigma = vitals[:, :, c].std() + 1e-8
        vitals[:, :, c] = (vitals[:, :, c] - mu) / sigma

    return vitals


def generate_imaging(X_tabular: np.ndarray, seed: int = config.SEED) -> np.ndarray:
    """
    Generate synthetic medical images from tabular features.

    Creates deterministic grayscale patterns seeded by patient features,
    simulating simplified chest X-ray-like structures.

    Parameters
    ----------
    X_tabular : (N, n_features) -- standardised tabular feature matrix
    seed      : random seed for reproducibility

    Returns
    -------
    images : (N, 1, H, W) float32 array
             N patients, 1 channel, IMG_SIZE × IMG_SIZE pixels, values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    N = X_tabular.shape[0]
    H = W = config.IMG_SIZE

    images = np.zeros((N, config.IMG_CHANNELS, H, W), dtype=np.float32)

    for i in range(N):
        # Create a base pattern influenced by patient features
        feat_hash = X_tabular[i, :min(6, X_tabular.shape[1])].sum()
        patient_rng = np.random.default_rng(int(abs(feat_hash * 1000)) % (2**31))

        # Simulate tissue density pattern
        base = patient_rng.uniform(0.2, 0.8, (H, W)).astype(np.float32)

        # Add circular structure (simulating cardiac region)
        yy, xx = np.ogrid[:H, :W]
        cx = H // 2 + int(np.clip(X_tabular[i, 0] * 2, -10, 10))
        cy = W // 2
        r = 12 + int(np.clip(abs(X_tabular[i, min(5, X_tabular.shape[1] - 1)]) * 3, 0, 15))
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
        base[mask] += 0.3

        # Normalise to [0, 1]
        bmin, bmax = base.min(), base.max()
        base = (base - bmin) / (bmax - bmin + 1e-8)
        images[i, 0] = base

    return images
