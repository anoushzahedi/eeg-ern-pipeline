"""
Preprocessing Module

This module handles raw EEG data preprocessing including filtering,
artifact removal, and ICA.
"""

import mne
import matplotlib
from .. import config
from .preprocessing import load_and_preprocess_subject_bids
from ..utils import save_checkpoint, find_sub_folders
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
        

def preprocess_subjects(subject_ids):
    """Preprocess specific subjects.

    Parameters
    ----------
    subject_ids : list of str
        List of subject IDs to process (without 'sub-' prefix)"""
    print(f"--- RUNNING PREPROCESSING ---")
    print(f"Data Root: {config.DATA_ROOT}")
    print(f"Processing {len(subject_ids)} specific subjects: {subject_ids}")
    setup_backend()
    mne.set_log_level("ERROR")
    for i, subject_id in enumerate(subject_ids, 1):
        print(f"\n[{i}/{len(subject_ids)}] Preprocessing: sub-{subject_id}")
        success = load_and_preprocess_subject_bids(
            bids_root=config.DATA_ROOT,
            subject=subject_id,
            task=config.TASK_NAME,
            **config.PREPRO_PARAMS,
        )
        if success:
            save_checkpoint(
                subject_id, config.PREPRO_PARAMS["pipeline_name"], config.DATA_ROOT
            )
    print("\n--- Preprocessing Complete ---")


def preprocess():
    """Preprocess all subjects (or test subset if TEST_MODE is enabled)."""
    print(f"--- RUNNING PREPROCESSING ---")
    print(f"Data Root: {config.DATA_ROOT}")
    setup_backend()
    mne.set_log_level("ERROR")
    subject_dirs = sorted(
        find_sub_folders(config.DATA_ROOT, expected_count=config.EXPECTED_SUBJECTS)
    )
    if config.TEST_MODE:
        subject_dirs = subject_dirs[config.NUM_SUB_TEST]
    for i, subject_path in enumerate(subject_dirs, 1):
        subject_id = subject_path.parent.name.replace("sub-", "")
        print(f"\n[{i}/{len(subject_dirs)}] Preprocessing: sub-{subject_id}")
        success = load_and_preprocess_subject_bids(
            bids_root=config.DATA_ROOT,
            subject=subject_id,
            task=config.TASK_NAME,
            **config.PREPRO_PARAMS,
        )
        if success:
            save_checkpoint(
                subject_id, config.PREPRO_PARAMS["pipeline_name"], config.DATA_ROOT
            )
    print("\n--- Preprocessing Complete ---")


if __name__ == "__main__":
    preprocess()
