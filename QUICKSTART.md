# Quick Start Guide

This guide will get you from zero to running the pipeline in ~10 minutes.

## Step 1: Environment Setup (Recommended)

We recommend using Conda to keep your environment clean:

Create a new environment:
```bash
conda create -n eeg_ern_pipeline python=3.10
conda activate eeg_ern_pipeline
```

## Step 2: Install the Pipeline

Download or clone this repository, enter the folder, and install it in **editable mode**:

```bash
pip install -e .
```

This command installs all dependencies and registers the `eeg_ern_pipe` command on your system.

## Step 3: Verify Your Data Structure

Your data should follow the BIDS-like structure:

```text
MyEEGData/
├── sub-001/
│   └── eeg/
│       └── sub-001_task-Flanker_eeg.set
├── sub-002/
│   └── eeg/
│       └── sub-002_task-Flanker_eeg.set
└── ...
```

## Step 4: Test the Pipeline

Run a quick test on 2 subjects. You can point to your data directly using the `--data_root` flag:

```bash
eeg_ern_pipe --all --test --data_root "/path/to/your/MyEEGData"
```

**This will:**
- Process the first 2 subjects.
- Run all 3 steps (preprocess → epoch → ERN).
- Take ~5-10 minutes depending on your computer.

## Step 5: Check the Results

Look in your data folder under `derivatives/`:

```text
derivatives/
├── preprocessing/     # ✓ Cleaned EEG files
├── epochs/           # ✓ Epoched data
└── ern/              # ✓ ERN results
    ├── sub-001/
    │   └── eeg/
    │       ├── sub-001_task-Flanker_ern.csv  # Individual results
    │       └── sub-001_task-Flanker_ern.png  # Plots
    ├── ern_amplitudes_FCz_summary.csv        # Summary for stats
    └── grand_average_task-Flanker_ern.png    # Group-level plot
```

## Step 6: Run on All Subjects

If the test was successful, run the full pipeline:

```bash
eeg_ern_pipe --all --data_root "/path/to/your/MyEEGData"
```

**Tip**: For 389 subjects, this may take 12-24 hours. Run it overnight!

---

## Summary of Commands

| Goal | Command |
|------|---------|
| **Install** | `pip install -e .` |
| **Test (2 subs)** | `eeg_ern_pipe --all --test --data_root "path"` |
| **Full Run** | `eeg_ern_pipe --all --data_root "path"` |
| **Interactive** | `eeg_ern_pipe --preprocess --interactive --subjects 001` |
| **Specific Subs** | `eeg_ern_pipe --all --subjects 001 002 003` |
| **Just ERN** | `eeg_ern_pipe --ern --data_root "path"` |

---

## Troubleshooting

### "Command not found: eeg_ern_pipe"
- Ensure you ran `pip install -e .` in the project root.
- Ensure your Conda environment is activated.

### "ModuleNotFoundError: No module named 'mne'"
- Run `pip install -r requirements.txt` or re-run `pip install -e .`.

### Plots are not showing
- Use the `--interactive` (or `--int`) flag.
- Note: Interactive mode is disabled by default for speed during large runs.

### Data Path Issues
- Always use absolute paths for `--data_root` to avoid confusion.
- Example: `"/Users/name/Documents/data"` instead of `"~/data"`.

---

## What's Next?

The main file for your statistics is:
`derivatives/ern/ern_amplitudes_FCz_summary.csv`

You can open this in **R**, **Python**, **SPSS**, or **Excel** to perform your group-level analysis.
