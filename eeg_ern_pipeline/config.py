from pathlib import Path
import os
import platform

DATA_ROOT = Path(os.getenv("EEG_DATA_ROOT", "./data"))

if not DATA_ROOT.exists():
    print(f"Warning: DATA_ROOT not found at {DATA_ROOT}. "
          "Please set the EEG_DATA_ROOT environment variable.")
EXPECTED_SUBJECTS = 389
TASK_NAME = "Flanker"
TEST_MODE = False
NUM_SUB_TEST = slice(None, 2)
INTERACTIVE_MODE = False
PREPRO_PARAMS = {
    "l_freq": 0.5,
    "h_freq": 40.0,
    "notch_freq": 50.0,
    "reference": "average",
    "ica_method": "picard",
    "ica_threshold": (3.0, 0.9),
    "bad_channel_threshold": 3.0,
    "interpolate_bads": True,
    "pipeline_name": "preprocessing",
}
EPOCH_PARAMS = {
    "tmin": -0.2,
    "tmax": 0.6,
    "baseline": (-0.2, 0),
    "use_autoreject": True,
    "n_jobs": -1,
    "output_pipeline": "epochs",
}
ERN_PARAMS = {
    "epochs_pipeline": "epochs",
    "ern_pipeline": "ern",
    "channel": "FCz",
    "ern_tmin": 0.0,
    "ern_tmax": 0.1,
    "use_peak": True,
    "save_plots": True,
    "merge_results": True,
    "output_filename": None,
}
