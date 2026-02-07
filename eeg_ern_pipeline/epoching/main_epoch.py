"""
Epoching Module

This module creates epochs from preprocessed continuous EEG data.
"""

import mne
import matplotlib
from pathlib import Path
from .. import config

if not config.INTERACTIVE_MODE:
    matplotlib.use("Agg")
from ..utils import find_sub_folders, save_checkpoint
from .epoching import epoch_and_save_bids_with_events


def epoch_subjects(subject_ids):
    """Epoch specific subjects.

    Parameters
    ----------
    subject_ids : list of str
        List of subject IDs to process (without 'sub-' prefix)"""
    print(f"--- EPOCHING ONLY ---")
    print(f"Note: Requires preprocessed files")
    print(f"Processing {len(subject_ids)} specific subjects: {subject_ids}")
    mne.set_log_level("ERROR")
    subjects_to_process = []
    for subject_id in subject_ids:
        preprocessed_file = (
            Path(config.DATA_ROOT)
            / "derivatives"
            / config.PREPRO_PARAMS["pipeline_name"]
            / f"sub-{subject_id}"
            / "eeg"
            / f"sub-{subject_id}_task-{config.TASK_NAME}_proc-clean_eeg.fif"
        )
        if preprocessed_file.exists():
            subjects_to_process.append(subject_id)
        else:
            print(f"Warning: No preprocessed file for sub-{subject_id}")
    for i, subject_id in enumerate(subjects_to_process, 1):
        print(f"\n[{i}/{len(subjects_to_process)}] Epoching: sub-{subject_id}")
        epoch_and_save_bids_with_events(
            bids_root=config.DATA_ROOT,
            subject=subject_id,
            task=config.TASK_NAME,
            **config.EPOCH_PARAMS,
        )
        save_checkpoint(
            subject_id, config.EPOCH_PARAMS["output_pipeline"], config.DATA_ROOT
        )
    print(f"\n--- Epoching Complete ---")
    print(f"Processed: {len(subjects_to_process)}/{len(subject_ids)} subjects")


def epoch():
    """Epoch all subjects (or test subset if TEST_MODE is enabled)."""
    print(f"--- EPOCHING ONLY ---")
    print(f"Note: Requires preprocessed files")
    mne.set_log_level("ERROR")
    subject_dirs = sorted(
        find_sub_folders(config.DATA_ROOT, expected_count=config.EXPECTED_SUBJECTS)
    )
    if config.TEST_MODE:
        subject_dirs = subject_dirs[config.NUM_SUB_TEST]
    subjects_to_process = []
    for subject_path in subject_dirs:
        subject_id = subject_path.parent.name.replace("sub-", "")
        preprocessed_file = (
            Path(config.DATA_ROOT)
            / "derivatives"
            / config.PREPRO_PARAMS["pipeline_name"]
            / f"sub-{subject_id}"
            / "eeg"
            / f"sub-{subject_id}_task-{config.TASK_NAME}_proc-clean_eeg.fif"
        )
        if preprocessed_file.exists():
            subjects_to_process.append(subject_id)
        else:
            print(f"Warning: No preprocessed file for sub-{subject_id}")
    for i, subject_id in enumerate(subjects_to_process, 1):
        print(f"\n[{i}/{len(subjects_to_process)}] Epoching: sub-{subject_id}")
        epoch_and_save_bids_with_events(
            bids_root=config.DATA_ROOT,
            subject=subject_id,
            task=config.TASK_NAME,
            **config.EPOCH_PARAMS,
        )
        save_checkpoint(
            subject_id, config.EPOCH_PARAMS["output_pipeline"], config.DATA_ROOT
        )
    print(f"\n--- Epoching Complete ---")
    print(f"Processed: {len(subjects_to_process)}/{len(subject_dirs)} subjects")


if __name__ == "__main__":
    epoch()
