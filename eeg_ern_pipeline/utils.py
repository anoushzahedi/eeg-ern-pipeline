import re
import json
from pathlib import Path
from datetime import datetime


def find_root(base_root: str = "."):
    base_root = Path(base_root)
    for path in base_root.iterdir():
        if path.is_dir() and re.match("^eeg", path.name, re.IGNORECASE):
            return path
    raise ValueError(
        'The current directory does not contain a folder starting with "eeg"'
    )


def find_sub_folders(root_path: Path, expected_count: int = 389):
    sub_dirs = []
    missing_eeg = []
    for path in root_path.iterdir():
        if re.match("^sub-[A-Z0-9]*", path.name):
            eeg_dir = path / "eeg"
            if eeg_dir.exists() and eeg_dir.is_dir():
                sub_dirs.append(eeg_dir)
            else:
                missing_eeg.append(path.name)
    print(f"Found {len(sub_dirs)} subject directories")
    if missing_eeg:
        print(f"Warning: {len(missing_eeg)} subjects missing 'eeg/' subdirectory")
    return sub_dirs


def save_checkpoint(subject_id: str, step: str, data_root: Path = None):
    """Save a checkpoint for a processing step.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., '01')
    step : str
        Processing step name (e.g., 'preprocessing', 'epoching')
    data_root : Path, optional
        BIDS root directory. If None, uses config.DATA_ROOT"""
    if data_root is None:
        from eeg_pipeline import config

        data_root = config.DATA_ROOT
    checkpoint_dir = data_root / "derivatives" / "eeg_pipeline" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"sub-{subject_id}_{step}.json"
    checkpoint_data = {
        "subject": f"sub-{subject_id}",
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
    }
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"  Checkpoint saved: {checkpoint_file.name}")
