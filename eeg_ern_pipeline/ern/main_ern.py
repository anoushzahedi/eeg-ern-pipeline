"""
ERN Analysis Module

This module extracts ERN/CRN amplitudes from epoched data.
"""

import mne
import matplotlib
from pathlib import Path
from .. import config
from ..utils import find_sub_folders, save_checkpoint
from .ern_analysis import process_subject_ern, merge_ern_results
import platform

def setup_backend():
    """Sets the backend based on the current config."""
    if not config.INTERACTIVE_MODE:
        matplotlib.use("Agg")
    else:
        if platform.system() == "Windows":
            try:
                matplotlib.use("Qt5Agg")
            except:
                pass
        elif platform.system() == "Darwin":
            matplotlib.use("MacOSX")

def ern_subjects(subject_ids):
    """Run ERN analysis on specific subjects.

    Parameters
    ----------
    subject_ids : list of str
        List of subject IDs to process (without 'sub-' prefix)"""
    print(f"--- RUNNING ERN ANALYSIS ---")
    print(
        f"Note: Requires epoched files from '{config.ERN_PARAMS['epochs_pipeline']}' pipeline"
    )
    print(f"Processing {len(subject_ids)} specific subjects: {subject_ids}")
    setup_backend()
    mne.set_log_level("ERROR")
    subjects_to_process = []
    for subject_id in subject_ids:
        epoched_file = (
            Path(config.DATA_ROOT)
            / "derivatives"
            / config.ERN_PARAMS["epochs_pipeline"]
            / f"sub-{subject_id}"
            / "eeg"
            / f"sub-{subject_id}_task-{config.TASK_NAME}_epo.fif"
        )
        if epoched_file.exists():
            subjects_to_process.append(subject_id)
        else:
            print(f"Warning: No epoched file for sub-{subject_id}")
    print(f"Found {len(subjects_to_process)} subjects with epoched data")
    successful = []
    failed = []
    for i, subject_id in enumerate(subjects_to_process, 1):
        print(f"\n[{i}/{len(subjects_to_process)}] Processing ERN: sub-{subject_id}")
        try:
            results = process_subject_ern(
                bids_root=config.DATA_ROOT,
                subject=subject_id,
                task=config.TASK_NAME,
                epochs_pipeline=config.ERN_PARAMS["epochs_pipeline"],
                ern_pipeline=config.ERN_PARAMS["ern_pipeline"],
                channel=config.ERN_PARAMS["channel"],
                ern_tmin=config.ERN_PARAMS["ern_tmin"],
                ern_tmax=config.ERN_PARAMS["ern_tmax"],
                use_peak=config.ERN_PARAMS["use_peak"],
                save_plots=config.ERN_PARAMS["save_plots"],
            )
            successful.append(subject_id)
            save_checkpoint(
                subject_id, config.ERN_PARAMS["ern_pipeline"], config.DATA_ROOT
            )
            print(f"  ✓ Successfully processed sub-{subject_id}")
        except Exception as e:
            failed.append(subject_id)
            print(f"  ✖ Error processing sub-{subject_id}: {e}")
            import traceback

            traceback.print_exc()
    print(f"\n{'=' * 60}")
    print("ERN PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Successful: {len(successful)}/{len(subjects_to_process)}")
    print(f"Failed: {len(failed)}/{len(subjects_to_process)}")
    if failed:
        print(f"\nFailed subjects: {', '.join(failed)}")
    if successful and config.ERN_PARAMS.get("merge_results", True):
        print(f"\n{'=' * 60}")
        print("MERGING RESULTS")
        print(f"{'=' * 60}")
        try:
            df = merge_ern_results(
                bids_root=config.DATA_ROOT,
                task=config.TASK_NAME,
                ern_pipeline=config.ERN_PARAMS["ern_pipeline"],
                output_filename=config.ERN_PARAMS.get("output_filename", None),
                channel=config.ERN_PARAMS["channel"],
            )
            print(f"\n✓ Successfully merged {len(df)} subjects")
        except Exception as e:
            print(f"\n✖ Error merging results: {e}")
            import traceback

            traceback.print_exc()
    print(f"\n--- ERN Analysis Complete ---")
    print(f"Processed: {len(subjects_to_process)}/{len(subject_ids)} subjects")


def ern():
    """Run ERN analysis on all subjects (or test subset if TEST_MODE is enabled)."""
    print(f"--- RUNNING ERN ANALYSIS ---")
    print(
        f"Note: Requires epoched files from '{config.ERN_PARAMS['epochs_pipeline']}' pipeline"
    )
    setup_backend()
    mne.set_log_level("ERROR")
    subject_dirs = sorted(
        find_sub_folders(config.DATA_ROOT, expected_count=config.EXPECTED_SUBJECTS)
    )
    if config.TEST_MODE:
        subject_dirs = subject_dirs[config.NUM_SUB_TEST]
    subjects_to_process = []
    for subject_path in subject_dirs:
        subject_id = subject_path.parent.name.replace("sub-", "")
        epoched_file = (
            Path(config.DATA_ROOT)
            / "derivatives"
            / config.ERN_PARAMS["epochs_pipeline"]
            / f"sub-{subject_id}"
            / "eeg"
            / f"sub-{subject_id}_task-{config.TASK_NAME}_epo.fif"
        )
        if epoched_file.exists():
            subjects_to_process.append(subject_id)
        else:
            print(f"Warning: No epoched file for sub-{subject_id}")
    print(f"Found {len(subjects_to_process)} subjects with epoched data")
    successful = []
    failed = []
    for i, subject_id in enumerate(subjects_to_process, 1):
        print(f"\n[{i}/{len(subjects_to_process)}] Processing ERN: sub-{subject_id}")
        try:
            results = process_subject_ern(
                bids_root=config.DATA_ROOT,
                subject=subject_id,
                task=config.TASK_NAME,
                epochs_pipeline=config.ERN_PARAMS["epochs_pipeline"],
                ern_pipeline=config.ERN_PARAMS["ern_pipeline"],
                channel=config.ERN_PARAMS["channel"],
                ern_tmin=config.ERN_PARAMS["ern_tmin"],
                ern_tmax=config.ERN_PARAMS["ern_tmax"],
                use_peak=config.ERN_PARAMS["use_peak"],
                save_plots=config.ERN_PARAMS["save_plots"],
            )
            successful.append(subject_id)
            save_checkpoint(
                subject_id, config.ERN_PARAMS["ern_pipeline"], config.DATA_ROOT
            )
            print(f"  ✓ Successfully processed sub-{subject_id}")
        except Exception as e:
            failed.append(subject_id)
            print(f"  ✖ Error processing sub-{subject_id}: {e}")
            import traceback

            traceback.print_exc()
    print(f"\n{'=' * 60}")
    print("ERN PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Successful: {len(successful)}/{len(subjects_to_process)}")
    print(f"Failed: {len(failed)}/{len(subjects_to_process)}")
    if failed:
        print(f"\nFailed subjects: {', '.join(failed)}")
    if successful and config.ERN_PARAMS.get("merge_results", True):
        print(f"\n{'=' * 60}")
        print("MERGING RESULTS")
        print(f"{'=' * 60}")
        try:
            df = merge_ern_results(
                bids_root=config.DATA_ROOT,
                task=config.TASK_NAME,
                ern_pipeline=config.ERN_PARAMS["ern_pipeline"],
                output_filename=config.ERN_PARAMS.get("output_filename", None),
                channel=config.ERN_PARAMS["channel"],
            )
            print(f"\n✓ Successfully merged {len(df)} subjects")
        except Exception as e:
            print(f"\n✖ Error merging results: {e}")
            import traceback

            traceback.print_exc()
    print(f"\n--- ERN Analysis Complete ---")
    print(f"Processed: {len(subjects_to_process)}/{len(subject_dirs)} subjects")


if __name__ == "__main__":
    ern()
