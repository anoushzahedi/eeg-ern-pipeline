import json
import pickle
import warnings
import traceback
from pathlib import Path
from typing import Optional, Tuple, Union, List
import numpy as np
import pandas as pd
import mne
from mne.report import Report
from mne_bids import BIDSPath, make_dataset_description
from autoreject import AutoReject


def load_bids_events_metadata(bids_root: Path, subject: str, task: str) -> tuple:
    """Load event information from BIDS events.json and events.tsv files.

    Parameters
    ----------
    bids_root : Path
        Root of the BIDS dataset.
    subject : str
        Subject ID (without 'sub-').
    task : str
        Task name.

    Returns
    -------
    event_mapping : dict
        Dictionary mapping event codes to event names.
    events_df : pd.DataFrame or None
        Events dataframe from events.tsv if it exists.
    events_metadata : dict
        Metadata from events.json."""
    events_json_path = (
        bids_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_events.json"
    )
    events_tsv_path = (
        bids_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_events.tsv"
    )
    event_mapping = {}
    events_df = None
    events_metadata = {}
    if events_json_path.exists():
        print(f"Loading events metadata from: {events_json_path}")
        with open(events_json_path, "r") as f:
            events_metadata = json.load(f)
        if "value" in events_metadata and "Levels" in events_metadata["value"]:
            levels = events_metadata["value"]["Levels"]
            for key, description in levels.items():
                if key.startswith("x"):
                    event_code = int(key[1:])
                    event_mapping[event_code] = description
                else:
                    event_code = int(key)
                    event_mapping[event_code] = description
        print(f"  Found {len(event_mapping)} event types in metadata")
        print(f"  Event codes: {sorted(event_mapping.keys())}")
    if events_tsv_path.exists():
        print(f"Loading events from: {events_tsv_path}")
        events_df = pd.read_csv(events_tsv_path, sep="\t")
        print(f"  Found {len(events_df)} events in TSV file")
    return (event_mapping, events_df, events_metadata)


def create_event_id_from_metadata(
    event_mapping: dict, filter_pattern: str = None
) -> dict:
    """Create event_id dictionary for MNE from BIDS event mapping.

    Parameters
    ----------
    event_mapping : dict
        Dictionary from load_bids_events_metadata().
    filter_pattern : str, optional
        Only include events containing this pattern (e.g., 'Target', 'Response_Correct').

    Returns
    -------
    event_id : dict
        Dictionary suitable for mne.Epochs(event_id=...)."""
    event_id = {}
    for code, description in event_mapping.items():
        if filter_pattern is None or filter_pattern in description:
            clean_name = description.replace(" ", "_").replace("/", "_")
            event_id[clean_name] = code
    return event_id


def map_annotations_to_event_codes(raw, event_mapping: dict) -> dict:
    """Create a mapping from MNE annotation descriptions to numeric event codes.

    EEGLAB stores events with string descriptions, but MNE converts them to
    annotations. This function maps those annotation descriptions back to
    the numeric codes defined in the BIDS events.json.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object with annotations from EEGLAB.
    event_mapping : dict
        Event code to description mapping from BIDS metadata.

    Returns
    -------
    annotation_to_code : dict
        Mapping from annotation description to event code."""
    annotation_descriptions = set(raw.annotations.description)
    description_to_code = {desc: code for code, desc in event_mapping.items()}
    annotation_to_code = {}
    for annot_desc in annotation_descriptions:
        if annot_desc in description_to_code:
            annotation_to_code[annot_desc] = description_to_code[annot_desc]
        else:
            for desc, code in description_to_code.items():
                if desc in annot_desc or annot_desc in desc:
                    annotation_to_code[annot_desc] = code
                    break
    return annotation_to_code


def remove_empty_epochs(
    epochs, variance_threshold=1e-12, min_active_channels=0.5, peak_to_peak_min=1e-06
):
    """Remove epochs that have no valid data (flat signals, all NaNs, etc.)

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to filter
    variance_threshold : float
        Minimum variance required per channel (default: 1e-12 V²)
    min_active_channels : float
        Minimum proportion of channels that must have variance above threshold (0-1)
    peak_to_peak_min : float
        Minimum peak-to-peak amplitude required (default: 1 µV)

    Returns
    -------
    epochs_valid : mne.Epochs
        Epochs with valid data only
    n_removed : int
        Number of empty epochs removed"""
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    valid_mask = np.ones(n_epochs, dtype=bool)
    for i in range(n_epochs):
        epoch_data = data[i]
        if np.all(np.isnan(epoch_data)):
            valid_mask[i] = False
            continue
        channel_variances = np.nanvar(epoch_data, axis=1)
        n_active_channels = np.sum(channel_variances >= variance_threshold)
        if n_active_channels < min_active_channels * n_channels:
            valid_mask[i] = False
            continue
        channel_ptp = np.nanmax(epoch_data, axis=1) - np.nanmin(epoch_data, axis=1)
        if np.median(channel_ptp) < peak_to_peak_min:
            valid_mask[i] = False
            continue
        nan_proportion = np.sum(np.isnan(epoch_data)) / epoch_data.size
        if nan_proportion > 0.5:
            valid_mask[i] = False
    epochs_valid = epochs[valid_mask]
    n_removed = n_epochs - len(epochs_valid)
    return (epochs_valid, n_removed)


def epoch_and_save_bids_with_events(
    bids_root: Path,
    subject: str,
    task: str = "Flanker",
    source_pipeline: str = "preprocessing",
    output_pipeline: str = "epochs",
    event_filter: str = None,
    specific_events: dict = None,
    tmin: float = -0.2,
    tmax: float = 0.8,
    baseline: tuple = (None, 0),
    use_autoreject: bool = True,
    n_interpolate: list = None,
    n_jobs: int = -1,
    random_state: int = 42,
    **epoch_kwargs,
) -> bool:
    """Load preprocessed BIDS data, create epochs using BIDS events metadata,
    apply autoreject, and save in BIDS format.

    Parameters
    ----------
    bids_root : Path
        Root of the BIDS dataset.
    subject : str
        Subject ID (without 'sub-').
    task : str
        Task name (default: 'Flanker').
    source_pipeline : str
        Source derivatives pipeline (default: 'preprocessing').
    output_pipeline : str
        Output derivatives pipeline (default: 'epochs').
    event_filter : str, optional
        Filter events by pattern (e.g., 'Target', 'Response_Correct').
    specific_events : dict, optional
        Manually specify event_id dictionary.
    tmin : float
        Start time before event (default: -0.2).
    tmax : float
        End time after event (default: 0.8).
    baseline : tuple
        Baseline correction window (default: (None, 0)).
    use_autoreject : bool
        Whether to use autoreject (default: True).
    n_interpolate : list, optional
        Number of channels for autoreject to interpolate.
    n_jobs : int
        Number of parallel jobs (default: -1).
    random_state : int
        Random state for autoreject (default: 42).
    **epoch_kwargs :
        Additional arguments for mne.Epochs().

    Returns
    -------
    bool
        True if successful, False otherwise."""
    print(f"\n{'=' * 60}")
    print(f"Epoching subject: sub-{subject}")
    print(f"{'=' * 60}")
    try:
        event_mapping, events_df, events_metadata = load_bids_events_metadata(
            bids_root, subject, task
        )
        if specific_events is not None:
            event_id = specific_events
            print(f"\nUsing manually specified events:")
        elif event_filter is not None:
            event_id = create_event_id_from_metadata(event_mapping, event_filter)
            print(f"\nFiltered events (pattern: '{event_filter}'):")
        else:
            event_id = create_event_id_from_metadata(event_mapping)
            print(f"\nUsing all events from metadata:")
        if not event_id:
            raise ValueError(f"No events found matching filter: {event_filter}")
        for name, code in sorted(event_id.items(), key=lambda x: x[1]):
            print(f"  {code}: {name}")
        source_root = bids_root / "derivatives" / source_pipeline
        preproc_bids_path = BIDSPath(
            subject=subject,
            task=task,
            datatype="eeg",
            processing="clean",
            suffix="eeg",
            extension=".fif",
            root=source_root,
            check=False,
        )
        print(f"\nLoading preprocessed data from: {preproc_bids_path.fpath}")
        if not preproc_bids_path.fpath.exists():
            raise FileNotFoundError(
                f"Preprocessed file not found: {preproc_bids_path.fpath}"
            )
        raw = mne.io.read_raw_fif(preproc_bids_path.fpath, preload=True)
        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.2f} seconds")
        print("\nExtracting events from annotations...")
        events, event_dict_from_annot = mne.events_from_annotations(raw)
        print(f"  Found {len(events)} total events")
        print(f"  Event types in annotations: {event_dict_from_annot}")
        annot_trigger_to_mne_code = event_dict_from_annot.copy()
        event_id_for_epochs = {}
        for event_name, original_trigger in event_id.items():
            trigger_str = str(original_trigger)
            if trigger_str in annot_trigger_to_mne_code:
                mne_code = annot_trigger_to_mne_code[trigger_str]
                event_id_for_epochs[event_name] = mne_code
        print(f"\n  Mapped event codes:")
        for name, code in sorted(event_id_for_epochs.items(), key=lambda x: x[1]):
            orig_trigger = event_id[name]
            print(f"    {name}: trigger {orig_trigger} -> MNE code {code}")
        unique_event_codes = np.unique(events[:, 2])
        print(f"\n  MNE event codes in data: {unique_event_codes}")
        missing_events = []
        for name, mne_code in event_id_for_epochs.items():
            if mne_code not in unique_event_codes:
                orig_trigger = event_id[name]
                missing_events.append(
                    f"{name} (trigger {orig_trigger}, MNE code {mne_code})"
                )
        if missing_events:
            print(f"  WARNING: Requested events not found in data: {missing_events}")
        print(f"\nCreating epochs (tmin={tmin}, tmax={tmax})...")
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id_for_epochs,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            **epoch_kwargs,
        )
        print(f"  Initial epochs: {len(epochs)}")
        print(f"  Epoch duration: {epochs.times[-1] - epochs.times[0]:.3f} seconds")
        print("\n  Epochs per condition:")
        for event_name in event_id_for_epochs.keys():
            try:
                count = len(epochs[event_name])
                print(f"    {event_name}: {count} epochs")
            except KeyError:
                print(f"    {event_name}: 0 epochs (not found)")
        reject_log = None
        ar = None
        if use_autoreject:
            print("\nApplying AutoReject...")
            print("  Pre-filtering empty/invalid epochs...")
            epochs_prefiltered, n_empty_removed = remove_empty_epochs(epochs)
            if n_empty_removed > 0:
                print(f"  Removed {n_empty_removed} empty/invalid epochs")
            if len(epochs_prefiltered) == 0:
                print("  CRITICAL: All epochs are empty/invalid.")
                print("  Skipping this subject.\n")
                return False
            if n_interpolate is None:
                n_interpolate = np.array([1, 2, 4, 8])
            ar = AutoReject(
                n_interpolate=n_interpolate,
                consensus=[0.1, 0.2],
                thresh_method="bayesian_optimization",
                cv=5,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                epochs_clean, reject_log = ar.fit_transform(
                    epochs_prefiltered, return_log=True
                )
            n_bad_epochs = np.sum(reject_log.bad_epochs)
            interpolated_counts = (reject_log.labels == 2).sum(axis=1)
            interpolated_epoch_indices = np.where(
                (interpolated_counts > 0) & ~reject_log.bad_epochs
            )[0]
            n_interpolated = len(interpolated_epoch_indices)
            print(f"  Epochs rejected by AutoReject: {n_bad_epochs}")
            print(f"  Epochs interpolated: {n_interpolated}")
            print(f"  Epochs remaining: {len(epochs_clean)}")
            if len(epochs_clean) == 0:
                print("\n  WARNING: AutoReject rejected 100% of epochs!")
                print("   Attempting recovery with fixed 300µV threshold...")
                reject_criteria = dict(eeg=0.0003)
                epochs_copy = epochs_prefiltered.copy()
                epochs_copy.drop_bad(reject=reject_criteria, verbose=False)
                if len(epochs_copy) == 0:
                    print("   CRITICAL: Even with 300µV threshold, 0 epochs remain.")
                    print("   Skipping this subject.\n")
                    return False
                else:
                    print(
                        f"   Recovery successful: {len(epochs_copy)} epochs kept with fallback method."
                    )
                    epochs_clean = epochs_copy
                    n_bad_epochs = n_empty_removed + (
                        len(epochs_prefiltered) - len(epochs_clean)
                    )
                    n_interpolated = 0
                    reject_log = None
            else:
                n_bad_epochs += n_empty_removed
            print("\n  Remaining epochs per condition:")
            for event_name in event_id_for_epochs.keys():
                try:
                    count = len(epochs_clean[event_name])
                    print(f"    {event_name}: {count} epochs")
                except KeyError:
                    print(f"    {event_name}: 0 epochs")
        else:
            print("\nSkipping autoreject (use_autoreject=False)")
            epochs_clean = epochs.copy()
            n_bad_epochs = 0
            n_interpolated = 0
        output_root = bids_root / "derivatives" / output_pipeline
        output_root.mkdir(parents=True, exist_ok=True)
        if not (output_root / "dataset_description.json").exists():
            make_dataset_description(
                path=output_root,
                name="EEG epoching pipeline",
                dataset_type="derivative",
                authors=["Anoushiravan Zahedi"],
                generated_by=[
                    {
                        "Name": output_pipeline,
                        "Description": "Epoched data with autoreject-based artifact removal",
                    }
                ],
                source_datasets=[{"URL": str(source_root)}],
            )
        epoch_bids_path = BIDSPath(
            subject=subject,
            task=task,
            datatype="eeg",
            suffix="epo",
            extension=".fif",
            root=output_root,
            check=False,
        )
        epoch_bids_path.fpath.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving epochs to: {epoch_bids_path.fpath}")
        epochs_clean.save(epoch_bids_path.fpath, overwrite=True)
        if use_autoreject and ar is not None and (reject_log is not None):
            ar_path = (
                epoch_bids_path.copy()
                .update(suffix="autoreject", extension=".pkl", check=False)
                .fpath
            )
            with open(ar_path, "wb") as f:
                pickle.dump(ar, f)
            print(f"  AutoReject object saved to: {ar_path}")
            log_path = (
                epoch_bids_path.copy()
                .update(suffix="rejlog", extension=".npz", check=False)
                .fpath
            )
            np.savez(
                log_path, bad_epochs=reject_log.bad_epochs, labels=reject_log.labels
            )
            print(f"  Rejection log saved to: {log_path}")
        print("\nGenerating HTML report...")
        report = Report(title=f"Epochs Report - sub-{subject}")
        report.add_epochs(epochs_clean, title="Cleaned Epochs", psd=True)
        if use_autoreject and reject_log is not None:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(
                ["Rejected", "Interpolated", "Good"],
                [n_bad_epochs, n_interpolated, len(epochs_clean)],
            )
            axes[0].set_ylabel("Number of epochs")
            axes[0].set_title("AutoReject Summary")
            axes[1].imshow(
                reject_log.labels.T,
                aspect="auto",
                cmap="RdYlGn_r",
                interpolation="nearest",
            )
            axes[1].set_xlabel("Epoch number")
            axes[1].set_ylabel("Channel")
            axes[1].set_title("Rejection pattern (0=good, 1=interpolated, 2=bad)")
            plt.tight_layout()
            report.add_figure(
                fig,
                title="AutoReject Results",
                caption=f"Rejected: {n_bad_epochs}, Interpolated: {n_interpolated}",
            )
            plt.close(fig)
        elif use_autoreject and reject_log is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.bar(["Rejected", "Good"], [n_bad_epochs, len(epochs_clean)])
            ax.set_ylabel("Number of epochs")
            ax.set_title("Fallback Rejection Summary (300µV threshold)")
            ax.text(
                0.5,
                0.95,
                "⚠️ AutoReject rejected 100% of epochs.\nFallback threshold (300µV) was used.",
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
            )
            plt.tight_layout()
            report.add_figure(
                fig,
                title="Fallback Rejection Results",
                caption=f"AutoReject failed. Used fixed 300µV threshold. Rejected: {n_bad_epochs}",
            )
            plt.close(fig)
        report_path = (
            epoch_bids_path.copy()
            .update(suffix="report", extension=".html", check=False)
            .fpath
        )
        report.save(report_path, overwrite=True, open_browser=False)
        print(f"  HTML report saved to: {report_path}")
        sidecar_json = (
            epoch_bids_path.copy().update(extension=".json", check=False).fpath
        )

        def make_json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            return obj

        metadata = {
            "Description": "Epoched EEG data with autoreject-based artifact removal",
            "TaskName": task,
            "SourcePipeline": source_pipeline,
            "EpochingPipeline": output_pipeline,
            "EventID": event_id,
            "EventID_MNE": event_id_for_epochs,
            "EventFilter": event_filter,
            "EventsMetadataSource": str(events_metadata) if events_metadata else None,
            "tmin": float(tmin),
            "tmax": float(tmax),
            "Baseline": baseline,
            "SamplingFrequency": float(epochs_clean.info["sfreq"]),
            "NumberOfChannels": int(len(epochs_clean.ch_names)),
            "InitialEpochs": int(len(epochs)),
            "FinalEpochs": int(len(epochs_clean)),
            "AutoRejectEnabled": use_autoreject,
            "Software": "MNE-Python + AutoReject",
            "MNEVersion": mne.__version__,
        }
        if use_autoreject and reject_log is not None:
            metadata["AutoRejectStats"] = {
                "EpochsRejected": int(n_bad_epochs),
                "EpochsInterpolated": int(n_interpolated),
                "RejectionRate": (
                    float(n_bad_epochs / len(epochs)) if len(epochs) > 0 else 0.0
                ),
            }
            metadata["EpochsPerCondition"] = {}
            for event_name in event_id_for_epochs.keys():
                try:
                    metadata["EpochsPerCondition"][event_name] = int(
                        len(epochs_clean[event_name])
                    )
                except KeyError:
                    metadata["EpochsPerCondition"][event_name] = 0
        elif use_autoreject and reject_log is None:
            metadata["AutoRejectStats"] = {
                "EpochsRejected": int(n_bad_epochs),
                "EpochsInterpolated": 0,
                "RejectionRate": (
                    float(n_bad_epochs / len(epochs)) if len(epochs) > 0 else 0.0
                ),
                "FallbackMethod": "Fixed 300µV threshold (AutoReject rejected 100%)",
            }
        metadata = make_json_serializable(metadata)
        with open(sidecar_json, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"  Metadata sidecar saved to: {sidecar_json}")
        print(f"\n✔ Successfully epoched sub-{subject}")
        return True
    except Exception as e:
        print(f"\n✖ ERROR epoching sub-{subject}: {e}")
        traceback.print_exc()
        return False
