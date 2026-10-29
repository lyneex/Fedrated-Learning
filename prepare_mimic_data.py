"""Build real linked MIMIC multimodal artifacts after credentialed download."""
from src.mimic_data import build_mimic_cohort, build_vitals_cache


if __name__ == "__main__":
    cohort = build_mimic_cohort()
    build_vitals_cache(cohort)
