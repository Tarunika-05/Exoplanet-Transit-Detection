"""
Central configuration for the Exoplanet Transit Detection Pipeline.
All targets, paths, and hyperparameters are defined here.
"""
import os
from pathlib import Path

# === PATHS ===
PROJECT_ROOT = Path(r"c:\Users\DELL\Desktop\Sangamithra Paper")
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for d in [DATA_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === TARGET STARS ===
TARGETS = [
    {
        "name": "Kepler-10",
        "planet": "Kepler-10b",
        "mission": "Kepler",
        "known_period": 0.837495,        # days
        "known_depth_ppm": 150,           # parts per million
        "known_duration_hours": 1.81,     # hours
        "rotation_period_days": 23.8,     # stellar rotation period
        "label": "confirmed",             # for ML training
    },
    {
        "name": "HAT-P-7",
        "planet": "HAT-P-7b",
        "mission": "Kepler",
        "known_period": 2.2047299,
        "known_depth_ppm": 6290,
        "known_duration_hours": 4.10,
        "rotation_period_days": None,     # unknown or not applicable
        "label": "confirmed",
    },
    {
        "name": "TrES-2",
        "planet": "TrES-2b",
        "mission": "Kepler",
        "known_period": 2.470614,
        "known_depth_ppm": 16500,
        "known_duration_hours": 1.77,
        "rotation_period_days": None,
        "label": "confirmed",
    },
    {
        "name": "Kepler-69",
        "planet": "Kepler-69c",
        "mission": "Kepler",
        "known_period": 242.4613,
        "known_depth_ppm": 200,
        "known_duration_hours": 9.6,
        "rotation_period_days": None,
        "label": "confirmed",
    },
    {
        "name": "Pi Mensae",
        "planet": "Pi Mensae c",
        "mission": "TESS",
        "known_period": 6.2679,
        "known_depth_ppm": 300,
        "known_duration_hours": 3.0,
        "rotation_period_days": None,
        "label": "confirmed",
    },
]

# === FFT PARAMETERS ===
FFT_LOW_FREQ_CUTOFF = 0.01       # cycles/day — remove frequencies below this
FFT_HIGH_FREQ_CUTOFF = None      # auto-compute from Nyquist
FFT_NOTCH_WIDTH = 0.005          # width of notch filter around stellar rotation harmonics
FFT_N_HARMONICS = 5              # number of rotation harmonics to remove

# === PREPROCESSING PARAMETERS ===
SIGMA_CLIP = 5.0                 # sigma for outlier removal
FLATTEN_WINDOW = 101             # Savitzky-Golay window for initial flattening

# === BLS PARAMETERS ===
BLS_PERIOD_MIN = 0.3             # days
BLS_PERIOD_MAX = 300.0           # days
BLS_N_PERIODS = 50000            # number of trial periods

# === ML PARAMETERS ===
ML_TEST_SIZE = 0.2               # train/test split
ML_RANDOM_STATE = 42
RF_N_ESTIMATORS = 200            # Random Forest trees
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 32
CNN_INPUT_LENGTH = 201           # number of flux points in phase-folded input

# === PLOTTING ===
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
MATPLOTLIB_STYLE = "seaborn-v0_8-darkgrid"
