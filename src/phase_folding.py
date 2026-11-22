import numpy as np

def characterize_transit(folded_lc, target_config):
    """Measures transit depth and duration from phase-folded light curve."""
    # Bin the data to reduce noise and get a clearer profile
    period = target_config.get("known_period", 1.0) # fallback
    
    # We want a reasonable number of bins, say 500 across the phase
    # Phase goes from -0.5 to 0.5, so total range is 1.0
    # Let's use a fixed number of bins for consistency
    bins = 500
    binned_lc = folded_lc.bin(bins=bins)
    
    # Remove NaNs from binned LC
    binned_lc = binned_lc.remove_nans()
    
    flux = binned_lc.flux.value
    phase = binned_lc.time.value # For folded LC, time is phase
    
    # Baseline flux (out of transit)
    # We assume transit is near phase 0.0, so baseline is near +/- 0.5
    out_of_transit = flux[np.abs(phase) > 0.1]
    if len(out_of_transit) > 0:
        baseline = np.median(out_of_transit)
    else:
        baseline = 1.0
        
    # Transit depth (minimum flux near phase 0.0)
    in_transit_mask = np.abs(phase) < 0.1
    if np.any(in_transit_mask):
        min_flux = np.min(flux[in_transit_mask])
        depth_rel = baseline - min_flux
        depth_ppm = depth_rel * 1e6
    else:
        depth_rel = 0.0
        depth_ppm = 0.0
        min_flux = 1.0
        
    # Estimate radius ratio Rp/Rs
    # Depth ≈ (Rp/Rs)^2
    if depth_rel > 0:
        rp_rs = np.sqrt(depth_rel)
    else:
        rp_rs = 0.0
        
    return {
        "measured_depth_ppm": depth_ppm,
        "rp_rs": rp_rs,
        "_folded_lc": folded_lc, # for plotting
        "_binned_lc": binned_lc
    }

def fold_all(filtered_dict, detections, targets):
    """Folds light curves and measures transit parameters."""
    results = []
    
    target_configs = {t['name']: t for t in targets}
    
    for det in detections:
        name = det['target_name']
        lc = filtered_dict.get(name)
        
        if lc is None:
            continue
            
        print(f"Phase folding {name}...")
        config = target_configs.get(name, {})
        
        # Use detected period and epoch
        period = det['detected_period']
        t0 = det['transit_epoch']
        
        # Fold the light curve
        # wrap_phase centers the transit at phase 0.0
        folded = lc.fold(period=period, epoch_time=t0, wrap_phase=period/2)
        
        # Normalize phase to [-0.5, 0.5] instead of [-period/2, period/2]
        # Lightkurve 'fold' with normalize_phase=True does this in newer versions
        # We'll just do it manually for safety
        folded.time = folded.time / period
        
        char_results = characterize_transit(folded, config)
        
        # Combine detection and characterization results
        full_result = {**det, **char_results}
        
        print(f"  Measured depth: {full_result['measured_depth_ppm']:.0f} ppm (Known: {config.get('known_depth_ppm', 'N/A')} ppm)")
        
        results.append(full_result)
        
    return results
