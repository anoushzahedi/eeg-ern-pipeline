# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-06

### Added
- **Grand average computation**: Automatic computation of grand averages across all subjects
- Grand average ERP plots showing group-level waveforms for all congruency conditions
- Grand average topography plots for group-level scalp distributions
- Saved grand average evoked objects (.fif files) for further analysis
- Support for four congruency levels (0%, 33%, 66%, 100%) in addition to overall averages
- Evoked object saving for each subject and condition to enable grand averaging
- Enhanced plotting with 2x2 subplot grid showing all congruency conditions
- BIDS dataset description generation for ERN derivative folder

### Changed
- Updated ERP plots to show all four congruency levels in a single 2x2 grid
- Improved plot aesthetics with better titles and formatting
- Enhanced summary statistics to include all congruency levels
- Streamlined dependencies (removed joblib - parallel processing removed)

### Fixed
- Plot layout improvements for better visualization of multiple conditions

### Migration Guide (1.0.0 → 1.1.0)
Version 1.1.0 is fully backward compatible. 
- **To Update**: Pull the latest code and run `python run_pipeline.py --ern`.
- **New Outputs**: You will automatically receive grand average plots and saved `.fif` evoked objects.
- **Customization**: To disable grand averaging in code, use `compute_grand_average=False` when calling `merge_ern_results`.

## New Analysis Possibilities

With evoked objects saved, you can now:

### 1. Custom Grand Averages

```python
import mne
from pathlib import Path

# Load specific evoked objects
evoked_files = Path('derivatives/ern').glob('sub-*/eeg/*_condition-Incorrect_0%-ave.fif')
evokeds = [mne.read_evokeds(f)[0] for f in evoked_files]

# Create custom grand average
grand_avg = mne.grand_average(evokeds)

# Custom plotting
grand_avg.plot_joint(times=[0.05, 0.08])
```

### 2. Statistical Comparison

```python
# Load grand averages
ga_incorrect = mne.read_evokeds('derivatives/ern/grand_average_task-Flanker_condition-Incorrect_all-ave.fif')[0]
ga_correct = mne.read_evokeds('derivatives/ern/grand_average_task-Flanker_condition-Correct_all-ave.fif')[0]

# Compute difference wave
diff_wave = mne.combine_evoked([ga_incorrect, ga_correct], weights=[1, -1])

# Plot
diff_wave.plot()
```

### 3. Cluster-based Permutation Tests

```python
# Use saved evoked objects for spatiotemporal cluster analysis
# (Example - would need additional setup)
from mne.stats import spatio_temporal_cluster_test

# ... load evoked objects for incorrect and correct ...
# ... run permutation test ...
```

### Planned
- Additional ERP components (P300, N200)
- Quality control metrics
- Enhanced statistical analysis integration

## [1.0.0] - 2026-01-10

### Added
- Complete EEG processing pipeline (preprocessing, epoching, ERN analysis)
- BIDS-compliant data organization
- Unified pipeline runner (`run_pipeline.py`)
- Subject-specific processing with `--subjects` flag
- Test mode for quick validation
- Automatic checkpointing for resumable processing
- Individual subject results and group-level summaries
- Comprehensive visualization (ERP plots and topographies)
- Configuration file for all analysis parameters
- Cleanup script for old ERN files
- Full documentation suite:
  - README.md (main documentation)
  - QUICKSTART.md (beginner guide)
  - INTEGRATION_GUIDE.md (detailed integration)
  - PIPELINE_USAGE_GUIDE.md (usage examples)
  - TEST_MODE_GUIDE.md (test vs. subject-specific)
  - REFACTORING_SUMMARY.md (migration guide)

### Changed
- Refactored ERN analysis into separate derivatives folder
- Improved error handling and reporting
- Separated processing and merging steps for better modularity

### Features
- Process all subjects or specific subjects
- Run complete pipeline or individual steps
- Test mode for development
- Flexible configuration system
- Comprehensive output structure

## [Unreleased]

### Planned
- Additional ERP component extraction (P300, N200, etc.)
- Grand average plots
- Statistical analysis integration
- Quality control metrics

---

## Version History

**[1.0.0]** - Initial release with complete pipeline

---

## How to Use This Changelog

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security fixes
