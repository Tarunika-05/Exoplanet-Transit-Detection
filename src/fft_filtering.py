import numpy as np
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks
import lightkurve as lk
from .config import FFT_LOW_FREQ_CUTOFF, FFT_NOTCH_WIDTH, FFT_N_HARMONICS

def identify_peaks(power_spectrum, xf, num_peaks=10):
    """Finds the most dominant frequency peaks in the power spectrum."""
    peaks, properties = find_peaks(power_spectrum, prominence=np.std(power_spectrum))
    if len(peaks) == 0:
        return []
        
    # Sort peaks by power (descending)
    peak_powers = power_spectrum[peaks]
    sorted_indices = np.argsort(peak_powers)[::-1]
    
    top_peaks = []
    for i in range(min(num_peaks, len(sorted_indices))):
        idx = peaks[sorted_indices[i]]
        freq = xf[idx]
        if freq > 0: # Avoid DC component
            period = 1.0 / freq
            top_peaks.append((freq, period, power_spectrum[idx]))
            
    return top_peaks

def design_filter(xf, target_config, sampling_interval):
    """Creates a frequency-domain mask to filter out noise."""
    mask = np.ones_like(xf)
    
    # 1. High-pass filter: remove very low frequencies (long-term drift)
    mask[xf < FFT_LOW_FREQ_CUTOFF] = 0.0
    
    # 2. Notch filter for stellar rotation (if known)
    rotation_period = target_config.get("rotation_period_days")
    known_period = target_config.get("known_period")
    
    if rotation_period is not None:
        rot_freq = 1.0 / rotation_period
        for harmonic in range(1, FFT_N_HARMONICS + 1):
            harmonic_freq = rot_freq * harmonic
            # Check if this harmonic is suspiciously close to the transit frequency
            # If so, don't notch it out, we don't want to kill the transit signal
            transit_freq = 1.0 / known_period
            
            # Simple check: if harmonic is within 10% of transit freq, skip
            if abs(harmonic_freq - transit_freq) / transit_freq < 0.1:
                print(f"  Warning: Harmonic {harmonic} ({harmonic_freq:.4f}) is close to transit freq ({transit_freq:.4f}). Skipping notch.")
                continue
                
            # Apply notch
            notch_idx = np.abs(xf - harmonic_freq) < FFT_NOTCH_WIDTH
            mask[notch_idx] = 0.0
            
    # 3. High-frequency noise removal (optional)
    # The Nyquist frequency is 1 / (2 * sampling_interval)
    # We can filter out frequencies very close to Nyquist if they are noisy
    nyquist = 1.0 / (2.0 * sampling_interval)
    mask[xf > 0.95 * nyquist] = 0.0
    
    return mask

def fft_filter_target(lc, target_config):
    """Applies FFT, filters the spectrum, and returns cleaned LightCurve."""
    flux = lc.flux.value
    time = lc.time.value
    n = len(flux)
    
    # Fill any remaining NaNs (should be none, but just in case for FFT)
    flux = np.nan_to_num(flux, nan=1.0)
    
    sampling_interval = np.median(np.diff(time)) # days
    
    # Forward FFT
    yf = rfft(flux)
    xf = rfftfreq(n, d=sampling_interval)
    power_spectrum = np.abs(yf) ** 2
    
    # Identify peaks (for logging/analysis)
    peaks = identify_peaks(power_spectrum, xf)
    if peaks:
        print(f"  Top frequency peak: {peaks[0][0]:.4f} cycles/day (Period: {peaks[0][1]:.2f} days)")
        
    # Design and apply filter
    filter_mask = design_filter(xf, target_config, sampling_interval)
    yf_filtered = yf * filter_mask
    
    # Inverse FFT
    flux_filtered = irfft(yf_filtered, n=n)
    
    # Reconstruct LightCurve object
    # irfft might return slightly different length if n is odd, but we provided n
    lc_filtered = lk.LightCurve(time=time, flux=flux_filtered)
    
    # Store FFT data in meta for visualization later
    lc_filtered.meta['xf'] = xf
    lc_filtered.meta['power_spectrum_raw'] = power_spectrum
    lc_filtered.meta['filter_mask'] = filter_mask
    lc_filtered.meta['power_spectrum_filtered'] = np.abs(yf_filtered) ** 2
    
    return lc_filtered

def fft_filter_all(cleaned_dict, targets):
    """Applies FFT filtering to all light curves."""
    filtered_dict = {}
    
    # Create target lookup
    target_configs = {t['name']: t for t in targets}
    
    for name, lc in cleaned_dict.items():
        print(f"Applying FFT filter to {name}...")
        config = target_configs.get(name, {})
        filtered_lc = fft_filter_target(lc, config)
        filtered_dict[name] = filtered_lc
        
    return filtered_dict
