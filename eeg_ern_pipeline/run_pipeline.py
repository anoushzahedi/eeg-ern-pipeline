"""
EEG Pipeline Runner

Run the complete EEG processing pipeline or individual steps.

Usage:
    # Run complete pipeline (all steps)
    eeg_ern_pipe --all

    # Run individual steps
    eeg_ern_pipe --preprocess
    eeg_ern_pipe --epoch
    eeg_ern_pipe --ern

    # Specify a custom data path (overrides config.DATA_ROOT)
    eeg_ern_pipe --all --data_root "/path/to/bids_data"

    # Run multiple specific steps in order
    eeg_ern_pipe --preprocess --epoch
    eeg_ern_pipe --epoch --ern

    # Run in test mode (uses TEST_MODE from config)
    eeg_ern_pipe --all --test
    eeg_ern_pipe --ern --test

    # Run in interactive mode (shows plots)
    eeg_ern_pipe --preprocess --interactive

    # Run for specific subjects only
    eeg_ern_pipe --all --subjects 001 002 003
    eeg_ern_pipe --ern --subjects 042

    # Combine options
    eeg_ern_pipe --epoch --ern --subjects 001 002 003 --data_root "./data"
"""

import argparse
import sys
from pathlib import Path
from . import config


def run_preprocessing(subject_ids=None):
    """Run preprocessing step."""
    from .preprocessing.main_preprocess import preprocess_subjects

    print("\n" + "=" * 60)
    print("STEP 1: PREPROCESSING")
    print("=" * 60)
    if subject_ids:
        preprocess_subjects(subject_ids)
    else:
        from .preprocessing.main_preprocess import preprocess

        preprocess()


def run_epoching(subject_ids=None):
    """Run epoching step."""
    from .epoching.main_epoch import epoch_subjects

    print("\n" + "=" * 60)
    print("STEP 2: EPOCHING")
    print("=" * 60)
    if subject_ids:
        epoch_subjects(subject_ids)
    else:
        from .epoching.main_epoch import epoch

        epoch()


def run_ern(subject_ids=None):
    """Run ERN analysis step."""
    from .ern.main_ern import ern_subjects

    print("\n" + "=" * 60)
    print("STEP 3: ERN ANALYSIS")
    print("=" * 60)
    if subject_ids:
        ern_subjects(subject_ids)
    else:
        from .ern.main_ern import ern

        ern()


def main():
    parser = argparse.ArgumentParser(
        description="Run EEG processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all processing steps in sequence"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Run preprocessing step only"
    )
    parser.add_argument("--epoch", action="store_true", help="Run epoching step only")
    parser.add_argument("--ern", action="store_true", help="Run ERN analysis step only")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Process specific subjects only (e.g., --subjects 001 002 003)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use TEST_MODE from config (overrides config.TEST_MODE)",
    )
    parser.add_argument(
        "--interactive",
        "--int",
        action="store_true",
        help="Use INTERACTIVE_MODE from config (overrides config.INTERACTIVE_MODE)",
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        help="Path to the BIDS root directory"
    )
    args = parser.parse_args()
    if args.data_root:
        config.DATA_ROOT = Path(args.data_root)
        print(f"The data path has been changed to {config.DATA_ROOT}")
    if args.test:
        config.TEST_MODE = True
        print(f"TEST MODE ENABLED: Will process {config.NUM_SUB_TEST}")
    if args.interactive:
        config.INTERACTIVE_MODE = True
        print(f"INTERACTIVE MODE ENABLED: Will show plots")
    if not (args.all or args.preprocess or args.epoch or args.ern):
        parser.print_help()
        print(
            "\nError: No processing step specified. Use --all or specify individual steps."
        )
        sys.exit(1)
    print("=" * 60)
    print("EEG PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"Data Root: {config.DATA_ROOT}")
    print(f"Task: {config.TASK_NAME}")
    print(f"Test Mode: {config.TEST_MODE}")
    if args.subjects:
        print(f"Subjects: {args.subjects}")
    else:
        print(f"Subjects: {('Test subset' if config.TEST_MODE else 'All available')}")
    if args.all:
        print(f"Steps: Preprocessing → Epoching → ERN Analysis")
    else:
        steps = []
        if args.preprocess:
            steps.append("Preprocessing")
        if args.epoch:
            steps.append("Epoching")
        if args.ern:
            steps.append("ERN Analysis")
        print(f"Steps: {' → '.join(steps)}")
    print("=" * 60)
    try:
        if args.all:
            run_preprocessing(args.subjects)
            run_epoching(args.subjects)
            run_ern(args.subjects)
        else:
            if args.preprocess:
                run_preprocessing(args.subjects)
            if args.epoch:
                run_epoching(args.subjects)
            if args.ern:
                run_ern(args.subjects)
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
