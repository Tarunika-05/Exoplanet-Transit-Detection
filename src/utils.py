import matplotlib.pyplot as plt
import pandas as pd
from src.config import FIGURES_DIR, RESULTS_DIR, FIGURE_DPI, FIGURE_FORMAT, MATPLOTLIB_STYLE

def setup_plotting():
    """Sets up publication-quality matplotlib parameters."""
    plt.style.use(MATPLOTLIB_STYLE)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 8),
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'savefig.format': FIGURE_FORMAT,
        'savefig.bbox': 'tight'
    })

def save_figure(fig, name):
    """Saves a figure to the FIGURES_DIR with configured format."""
    filepath = FIGURES_DIR / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(filepath)
    print(f"Saved figure: {filepath.name}")

def make_results_table(results_list):
    """Converts a list of result dictionaries to a pandas DataFrame and saves as CSV."""
    df = pd.DataFrame(results_list)
    csv_path = RESULTS_DIR / "pipeline_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path.name}")
    return df

def print_summary(results_df):
    """Pretty-prints the summary results."""
    print("\n" + "="*60)
    print("PIPELINE RESULTS SUMMARY")
    print("="*60)
    
    # Check if ML columns exist
    if 'rf_accuracy' in results_df.columns:
        cols_to_print = ['target_name', 'known_period', 'detected_period', 'period_error_pct', 'rf_accuracy']
    else:
        cols_to_print = ['target_name', 'known_period', 'detected_period', 'period_error_pct', 'sde']
        
    print(results_df[cols_to_print].to_string(index=False))
    print("="*60)
