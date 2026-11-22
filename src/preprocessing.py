import numpy as np
from .config import SIGMA_CLIP, FLATTEN_WINDOW

def preprocess_target(lc):
    """Cleans and normalizes a single light curve."""
    if lc is None:
        return None
        
    # Remove NaN values (which are common in TESS/Kepler data)
    lc_clean = lc.remove_nans()
    
    # Remove large outliers using sigma clipping
    lc_clean = lc_clean.remove_outliers(sigma=SIGMA_CLIP)
    
    # Normalize flux so the median is 1.0 (standard for exoplanet work)
    lc_clean = lc_clean.normalize()
    
    # Optional: We could apply a Savitzky-Golay filter here to flatten
    # long-term trends before FFT, but the FFT high-pass filter will
    # handle low-frequency drift. Let's rely on the FFT for detrending.
    
    return lc_clean

def preprocess_all(data_dict):
    """Preprocesses all light curves in the dictionary."""
    cleaned_dict = {}
    for name, lc in data_dict.items():
        print(f"Preprocessing {name}...")
        cleaned_lc = preprocess_target(lc)
        if cleaned_lc is not None:
            cleaned_dict[name] = cleaned_lc
            
    return cleaned_dict
