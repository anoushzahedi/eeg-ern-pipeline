# EEG Pipeline for ERN Analysis

A complete Python pipeline for processing EEG data and extracting Error-Related Negativity (ERN) amplitudes from Flanker task data. This pipeline follows BIDS-derivatives conventions and provides a modular, reproducible workflow.

## Features

- **Professional CLI**: Run the entire pipeline via the `eeg_ern_pipe` command.
- **Modular Architecture**: Organized sub-packages for Preprocessing, Epoching, and ERN Analysis.
- **BIDS-compliant**: Follows BIDS-derivatives standard for data organization.
- **Flexible Execution**: Override data paths and parameters via command-line arguments.
- **Checkpointing**: Automatic progress tracking for resuming interrupted runs.
- **Grand Averages**: Automatic computation of group-level ERPs and topographies.
- **Interactive Mode**: Toggle visualizations on/off for speed or inspection.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Installation

### For Users
To install the latest version directly from GitHub:
```bash
pip install git+https://github.com/anoushzahedi/eeg-ern-pipeline.git
```

### For Developers
If you plan to modify the code, clone the repository and install it in **editable mode**:
```bash
git clone https://github.com/anoushzahedi/eeg-ern-pipeline.git
cd eeg-ern-pipeline
pip install -e .
```

## Quick Start

### 1. Test the Pipeline
Validate the setup on a small subset of subjects (first 2 by default):
```bash
eeg_ern_pipe --all --test --data_root "/path/to/your/data"
```

### 2. Run Full Pipeline
Process all subjects in your dataset:
```bash
eeg_ern_pipe --all --data_root "/path/to/your/data"
```

## Configuration

While most settings can be handled via the CLI, advanced analysis parameters are located in `eeg_ern_pipeline/config.py`.

### Key Parameters
- **PREPRO_PARAMS**: Filter settings (0.5-40Hz), notch frequency, and ICA settings.
- **EPOCH_PARAMS**: Time windows (-0.2 to 0.6s), baseline correction, and AutoReject thresholds.
- **ERN_PARAMS**: Analysis channel (default: `FCz`), peak vs. mean amplitude, and time windows.

## Usage

### Command-Line Interface (CLI)
The `eeg_ern_pipe` command is the primary way to interact with the pipeline.

```bash
# Run everything
eeg_ern_pipe --all --data_root "./data"

# Run individual steps
eeg_ern_pipe --preprocess
eeg_ern_pipe --epoch
eeg_ern_pipe --ern

# Run with plots (Interactive Mode)
eeg_ern_pipe --preprocess --interactive --subjects 001

# Process specific subjects
eeg_ern_pipe --all --subjects 001 005 010
```

### Python API
You can also use the sub-packages programmatically:

```python
from eeg_ern_pipeline import config
from eeg_ern_pipeline.preprocessing import main_preprocess
from eeg_ern_pipeline.epoching import main_epoch
from eeg_ern_pipeline.ern import main_ern

# Example: Run preprocessing for a list of subjects
subjects = ['001', '002']
main_preprocess.preprocess_subjects(subjects)
```

## Directory Structure

```text
eeg-ern-pipeline/
├── pyproject.toml             # Package metadata and entry points
├── QUICKSTART.md              # Fast-track guide
├── README.md                  # This file
├── eeg_ern_pipeline/          # Main Package
│   ├── __init__.py            # Package exports
│   ├── config.py              # Global configuration
│   ├── run_pipeline.py        # CLI Logic
│   ├── utils.py               # Shared utilities
│   ├── preprocessing/         # Preprocessing Sub-package
│   │   ├── preprocessing.py   # Logic
│   │   └── main_preprocess.py # Runner
│   ├── epoching/              # Epoching Sub-package
│   │   ├── epoching.py
│   │   └── main_epoch.py
│   └── ern/                   # ERN Sub-package
│       ├── ern_analysis.py
│       └── main_ern.py
```

## Output Structure

Results are saved in the `derivatives/` folder of your dataset:

```text
derivatives/
├── preprocessing/             # Cleaned .fif files
├── epochs/                    # Epoched .fif files
└── ern/                       # Analysis Results
    ├── sub-*/                 # Individual CSVs and PNG plots
    ├── ern_summary.csv        # Group-level summary for statistics
    └── grand_average_*.png    # Group-level ERP visualizations
```

## Troubleshooting

- **Memory Issues**: If the pipeline freezes during the "Merging Results" step with 300+ subjects, ensure you are not using the `--interactive` flag.
- **Path Errors**: Always use absolute paths for the `--data_root` argument.
- **Backend Errors**: If plots don't appear in interactive mode, ensure you have a GUI backend installed (e.g., `PyQt5`).

## License
This project is licensed under the MIT License.

## Support
For bugs or feature requests, please open an issue on the [GitHub repository](https://github.com/anoushzahedi/eeg-ern-pipeline).
