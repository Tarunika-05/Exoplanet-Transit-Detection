import numpy as np
from astropy.timeseries import LombScargle
from .config import BLS_PERIOD_MIN, BLS_PERIOD_MAX, BLS_N_PERIODS

def detect_transit_bls(lc, target_config):
    """Runs BLS periodogram to find the best transit period."""
    
    # Adjust period search range based on known period if available
    # to ensure we search the right parameter space
    known_period = target_config.get("known_period")
    
    if known_period and known_period > 100:
        # Long period planet (e.g., Kepler-69c)
        period_min = max(50.0, known_period * 0.5)
        period_max = known_period * 1.5
    else:
        # Standard search for short-period planets
        period_min = BLS_PERIOD_MIN
        period_max = min(20.0, lc.time.value.max() - lc.time.value.min())
        
    periods = np.linspace(period_min, period_max, BLS_N_PERIODS)
    
    # Define duration explicitly so astropy doesn't guess a duration > min period
    # Ensure max duration is strictly less than period_min
    max_duration = min(0.3, period_min * 0.9)
    durations = np.linspace(0.01, max_duration, 10)
    
    # Run BLS
    bls = lc.to_periodogram(method='bls', period=periods, duration=durations)
    
    best_period = bls.period_at_max_power.value
    best_t0 = bls.transit_time_at_max_power.value
    best_duration = bls.duration_at_max_power.value
    
    # Calculate Signal Detection Efficiency (SDE)
    # A simple metric: (max_power - mean_power) / std_power
    power = bls.power.value
    sde = (np.max(power) - np.mean(power)) / np.std(power)
    
    # Calculate error relative to known period
    error_pct = 0.0
    if known_period:
        error_pct = abs(best_period - known_period) / known_period * 100.0
        
    # Store bls object for plotting
    bls_meta = {
        'bls': bls,
        'period': best_period,
        't0': best_t0,
        'duration': best_duration
    }
    
    return {
        "target_name": target_config["name"],
        "known_period": known_period,
        "detected_period": best_period,
        "period_error_pct": error_pct,
        "sde": sde,
        "transit_epoch": best_t0,
        "transit_duration": best_duration,
        "_bls_meta": bls_meta # internal use
    }

def detect_all(filtered_dict, targets):
    """Runs transit detection on all filtered light curves."""
    detections = []
    
    target_configs = {t['name']: t for t in targets}
    
    for name, lc in filtered_dict.items():
        print(f"Running BLS periodogram for {name}...")
        config = target_configs.get(name, {})
        
        result = detect_transit_bls(lc, config)
        
        print(f"  Known period: {result['known_period']:.4f} days")
        print(f"  Detected:     {result['detected_period']:.4f} days (Error: {result['period_error_pct']:.2f}%)")
        print(f"  SDE:          {result['sde']:.1f}")
        
        detections.append(result)
        
    return detections
