from . import config
from .utils import find_root, find_sub_folders, save_checkpoint
from .preprocessing.preprocessing import (
    preprocess_eeg,
    load_and_preprocess_subject_bids
)

from .epoching.epoching import (
    epoch_and_save_bids_with_events
)

from .ern.ern_analysis import (
    process_subject_ern,
    merge_ern_results
)
from .preprocessing import main_preprocess
from .epoching import main_epoch
from .ern import main_ern

__version__ = "1.1.0"
