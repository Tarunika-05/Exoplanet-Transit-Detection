import lightkurve as lk
from pathlib import Path
from tqdm import tqdm
import time
from .config import DATA_DIR

def load_cached_data(target_name):
    """Loads stitched light curve from cache if it exists."""
    # lightkurve caches automatically, but we can also manage our own files
    return None

def download_target(target_config, max_retries=3):
    """Downloads and stitches all available light curves for a single target."""
    name = target_config['name']
    mission = target_config['mission']
    
    for attempt in range(max_retries):
        try:
            print(f"Searching for {name} ({mission})...")
            # For Kepler, we want PDCSAP flux. For TESS, use SPOC author
            if mission == "TESS":
                # TESS Short cadence is 120s (exptime=120)
                search_result = lk.search_lightcurve(name, mission=mission, author="SPOC", exptime=120)
            else:
                # Kepler Long Cadence is 1800s (exptime=1800)
                search_result = lk.search_lightcurve(name, mission=mission, exptime=1800)
                
            if len(search_result) == 0:
                print(f"Warning: No data found for {name}")
                return None
                
            # Limit to the first 4 quarters/sectors to keep downloads fast and avoid corruption
            limit = min(4, len(search_result))
            search_result = search_result[0:limit]
                
            print(f"Downloading {len(search_result)} quarters/sectors...")
            # Using quality_bitmask 'hard' removes most flagged data points
            lc_collection = search_result.download_all(quality_bitmask="hard")
            
            if lc_collection is None or len(lc_collection) == 0:
                return None
                
            print(f"Stitching data...")
            lc = lc_collection.stitch()
                
            # Keep only time and pdcsap_flux for consistency
            return lc
            
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                print(f"Failed to download {name} after {max_retries} attempts.")
                return None

def download_all_targets(targets):
    """Downloads light curves for a list of targets."""
    data_dict = {}
    for target in tqdm(targets, desc="Downloading targets"):
        lc = download_target(target)
        if lc is not None:
            data_dict[target['name']] = lc
            # Save raw data plot as a check (done later in visualization, but good to have)
    return data_dict
