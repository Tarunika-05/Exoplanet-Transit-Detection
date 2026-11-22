import matplotlib.pyplot as plt
import numpy as np
from .utils import save_figure

def plot_lightcurve(lc, ax, title="Light Curve", color='k'):
    """Helper to plot a light curve on a specific axis."""
    if lc is None:
        return
    
    time = lc.time.value
    flux = lc.flux.value
    
    ax.scatter(time, flux, s=1, color=color, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized Flux")

def plot_power_spectrum(lc_filtered, ax, title="Power Spectrum"):
    """Plots the power spectrum and the filter mask."""
    if lc_filtered is None or 'xf' not in lc_filtered.meta:
        return
        
    xf = lc_filtered.meta['xf']
    raw_power = lc_filtered.meta['power_spectrum_raw']
    filtered_power = lc_filtered.meta['power_spectrum_filtered']
    mask = lc_filtered.meta['filter_mask']
    
    # Plot positive frequencies only
    pos_idx = xf > 0
    
    ax.plot(xf[pos_idx], raw_power[pos_idx], color='grey', alpha=0.5, label='Raw')
    ax.plot(xf[pos_idx], filtered_power[pos_idx], color='b', alpha=0.8, label='Filtered')
    
    # Overlay mask (scaled for visibility)
    ax.plot(xf[pos_idx], mask[pos_idx] * np.max(raw_power[pos_idx]), color='r', linestyle='--', alpha=0.5, label='Filter Mask')
    
    ax.set_yscale('log')
    ax.set_xlim(0, 30) # Zoom in on interesting frequencies
    ax.set_title(title)
    ax.set_xlabel("Frequency (cycles/day)")
    ax.set_ylabel("Power")
    ax.legend(loc='upper right')

def plot_phase_folded(det, ax, title="Phase Folded Transit"):
    """Plots the phase-folded and binned transit."""
    folded = det.get('_folded_lc')
    binned = det.get('_binned_lc')
    
    if folded is None or binned is None:
        ax.text(0.5, 0.5, "Phase Fold Failed\n(No Transit Detected)", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, color='red', fontsize=12)
        ax.set_title(title)
        ax.set_axis_off()
        return
        
    # Scatter folded points
    ax.scatter(folded.time.value, folded.flux.value, s=1, color='grey', alpha=0.2)
    
    # Plot binned points
    ax.plot(binned.time.value, binned.flux.value, 'ko', markersize=3, label='Binned Data')
    
    ax.set_xlim(-0.1, 0.1) # Zoom on transit
    ax.set_title(title)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalized Flux")

def generate_all_figures(data, cleaned, filtered, detections, results_df, targets):
    """Generates the publication-quality summary figures."""
    print("Generating figures...")
    
    n_targets = len([t for t in targets if t['name'] in data])
    if n_targets == 0:
        print("No data available to plot.")
        return
        
    # 1. Summary All Targets (Raw | Filtered | Folded)
    fig1, axes1 = plt.subplots(n_targets, 3, figsize=(15, 3 * n_targets))
    if n_targets == 1:
        axes1 = [axes1] # Ensure 2D array
        
    row = 0
    for det in detections:
        name = det['target_name']
        lc_raw = data.get(name)
        lc_filt = filtered.get(name)
        
        if lc_raw is None:
            continue
            
        # Raw
        plot_lightcurve(lc_raw, axes1[row][0], title=f"{name} (Raw)", color='grey')
        
        # Filtered
        plot_lightcurve(lc_filt, axes1[row][1], title=f"{name} (FFT Filtered)", color='blue')
        
        # Folded
        plot_phase_folded(det, axes1[row][2], title=f"{name} (Phase Folded)")
        
        row += 1
        
    plt.tight_layout()
    save_figure(fig1, "summary_all_targets")
    plt.close(fig1)
    
    # 2. Power Spectra
    fig2, axes2 = plt.subplots(n_targets, 1, figsize=(10, 3 * n_targets))
    if n_targets == 1:
        axes2 = [axes2]
        
    row = 0
    for det in detections:
        name = det['target_name']
        lc_filt = filtered.get(name)
        
        if lc_filt is not None:
            plot_power_spectrum(lc_filt, axes2[row], title=f"{name} Power Spectrum")
            row += 1
            
    plt.tight_layout()
    save_figure(fig2, "summary_power_spectra")
    plt.close(fig2)
    
    # 3. Detection Results Bar Chart
    if not results_df.empty:
        fig3, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results_df))
        width = 0.35
        
        ax.bar(x - width/2, results_df['known_period'], width, label='Known Period')
        ax.bar(x + width/2, results_df['detected_period'], width, label='Detected Period')
        
        ax.set_ylabel('Period (days)')
        ax.set_title('Known vs Detected Orbital Periods')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['target_name'])
        ax.legend()
        
        # Add error percentages as text
        for i, row_data in results_df.iterrows():
            ax.text(i, max(row_data['known_period'], row_data['detected_period']) + 0.5, 
                    f"Err: {row_data['period_error_pct']:.2f}%", 
                    ha='center')
                    
        plt.tight_layout()
        save_figure(fig3, "summary_detection_results")
        plt.close(fig3)
