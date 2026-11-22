"""
Exoplanet Transit Detection Pipeline
=====================================
Applies FFT-based noise filtering to Kepler/TESS light curves
to detect and characterize transiting exoplanets.

Usage:
    python main.py                    # Run full pipeline
    python main.py --targets 0 1 2    # Run for specific target indices only
    python main.py --skip-ml          # Skip ML classification phase
    python main.py --skip-cnn         # Skip CNN (use only Random Forest)
"""
import argparse
from src.config import TARGETS, FIGURES_DIR, RESULTS_DIR
from src.data_acquisition import download_all_targets
from src.preprocessing import preprocess_all
from src.fft_filtering import fft_filter_all
from src.transit_detection import detect_all
from src.phase_folding import fold_all
from src.ml_classifier import run_ml_pipeline
from src.visualization import generate_all_figures
from src.utils import setup_plotting, make_results_table, print_summary

def main(args):
    setup_plotting()
    
    # Phase 1: Data Acquisition
    print("=" * 60)
    print("PHASE 1: Data Acquisition")
    print("=" * 60)
    targets = [TARGETS[i] for i in args.targets] if args.targets else TARGETS
    data = download_all_targets(targets)
    
    # Phase 2: Preprocessing
    print("\n" + "=" * 60)
    print("PHASE 2: Preprocessing")
    print("=" * 60)
    cleaned = preprocess_all(data)
    
    # Phase 3: FFT Filtering
    print("\n" + "=" * 60)
    print("PHASE 3: FFT Noise Filtering")
    print("=" * 60)
    filtered = fft_filter_all(cleaned, targets)
    
    # Phase 4: Transit Detection
    print("\n" + "=" * 60)
    print("PHASE 4: Transit Detection (BLS)")
    print("=" * 60)
    detections = detect_all(filtered, targets)
    
    # Phase 5: Phase Folding
    print("\n" + "=" * 60)
    print("PHASE 5: Phase Folding & Characterization")
    print("=" * 60)
    results = fold_all(filtered, detections, targets)
    
    # Phase 6: ML Classification
    if not args.skip_ml:
        print("\n" + "=" * 60)
        print("PHASE 6: ML Classification")
        print("=" * 60)
        ml_results = run_ml_pipeline(filtered, targets, skip_cnn=args.skip_cnn)
        # Update results with ML metrics
        results = [{**r, **ml} for r, ml in zip(results, ml_results)]
    
    # Phase 7: Results
    print("\n" + "=" * 60)
    print("PHASE 7: Results Summary")
    print("=" * 60)
    results_df = make_results_table(results)
    generate_all_figures(data, cleaned, filtered, results, results_df, targets)
    print_summary(results_df)
    
    print(f"\n✅ Pipeline complete! Figures saved to: {FIGURES_DIR}")
    print(f"   Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exoplanet Transit Detection Pipeline")
    parser.add_argument("--targets", nargs="+", type=int, help="Target indices to process (0-4)")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML classification")
    parser.add_argument("--skip-cnn", action="store_true", help="Skip CNN, use only Random Forest")
    args = parser.parse_args()
    main(args)
