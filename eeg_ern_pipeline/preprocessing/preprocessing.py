import json
import warnings
import traceback
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.report import Report
from mne_bids import BIDSPath, make_dataset_description
from .. import config
import matplotlib.pyplot as plt


def identify_flat_segments(
    raw,
    win_size=2.0,
    step_size=1.0,
    variance_threshold=1e-12,
    min_active_channels=0.5,
    peak_to_peak_min=1e-06,
):
    """Identify flat/empty segments in continuous raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous raw data
    win_size : float
        Window size in seconds for checking segments
    step_size : float
        Step size in seconds (sliding window)
    variance_threshold : float
        Minimum variance required per channel
    min_active_channels : float
        Minimum proportion of channels that must be active (0-1)
    peak_to_peak_min : float
        Minimum peak-to-peak amplitude required

    Returns
    -------
    annotations : mne.Annotations or None
        Annotations marking flat segments as 'BAD_flat'"""
    from mne import Annotations

    sfreq = raw.info["sfreq"]
    n_samples_win = int(win_size * sfreq)
    n_samples_step = int(step_size * sfreq)
    data = raw.get_data(picks="eeg")
    n_channels, n_times = data.shape
    onsets = []
    durations = []
    start_idx = 0
    while start_idx + n_samples_win <= n_times:
        segment = data[:, start_idx : start_idx + n_samples_win]
        is_flat = False
        channel_variances = np.var(segment, axis=1)
        n_active = np.sum(channel_variances >= variance_threshold)
        if n_active < min_active_channels * n_channels:
            is_flat = True
        if not is_flat:
            channel_ptp = np.max(segment, axis=1) - np.min(segment, axis=1)
            if np.median(channel_ptp) < peak_to_peak_min:
                is_flat = True
        if is_flat:
            onset_sec = start_idx / sfreq
            onsets.append(onset_sec)
            durations.append(win_size)
            start_idx += n_samples_win
        else:
            start_idx += n_samples_step
    if onsets:
        merged_onsets = [onsets[0]]
        merged_durations = [durations[0]]
        for i in range(1, len(onsets)):
            last_end = merged_onsets[-1] + merged_durations[-1]
            current_start = onsets[i]
            if current_start <= last_end + 0.1:
                merged_durations[-1] = current_start + durations[i] - merged_onsets[-1]
            else:
                merged_onsets.append(current_start)
                merged_durations.append(durations[i])
        annotations = Annotations(
            onset=merged_onsets,
            duration=merged_durations,
            description=["BAD_flat"] * len(merged_onsets),
        )
        return annotations
    return None


def preprocess_eeg(
    raw: mne.io.Raw,
    l_freq: float,
    h_freq: float,
    notch_freq: float,
    reference: str,
    ica_n_components: Optional[float] = 0.999,
    ica_method: str = "picard",
    ica_threshold: Optional[tuple] = (2.5, 0.8),
    bad_channel_threshold: float = 3.0,
    interpolate_bads: bool = False,
    manual_ica_detection: bool = False,
):
    """Apply a standard EEG preprocessing pipeline, including ICA and bad-channel detection.

    Args:
        raw (mne.io.Raw): Raw EEG data
        l_freq (float): Low-pass filter frequency
        h_freq (float): High-pass filter frequency
        notch_freq (float): Notch filter frequency for line noise
        reference (str): Reference type: 'average', 'REST', or list of channel names
        ica_n_components (float): Number of ICA components; default: .999
        ica_method (str): ICA method: 'fastica', 'infomax', or 'picard' (default: 'picard')
        ica_threshold (tuple): Thresholds for ICA detection (zscore, correlation)
        bad_channel_threshold (float): Z-score threshold for bad channel detection
        interpolate_bads (bool): Whether to interpolate bad channels
        manual_ica_detection (bool): only for debugging

    Returns:
        tuple: (preprocessed raw, bad_channels, n_ica_components_removed, ica_object, eog_evoked)
    """
    print("  Starting preprocessing...")
    print(f"  - Applying band-pass filter: {l_freq}-{h_freq} Hz")
    raw.filter(l_freq, h_freq, fir_design="firwin")
    if notch_freq is not None:
        print(f"  - Applying notch filter at {notch_freq} Hz")
        raw.notch_filter(notch_freq, picks="eeg")
    print(f"  - Detecting bad channels (threshold: {bad_channel_threshold} SD)")
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    if len(eeg_picks) > 0:
        data = raw.get_data(picks=eeg_picks)
        channel_std = np.std(data, axis=1)
        channel_mean = np.mean(np.abs(data), axis=1)
        std_z = np.abs((channel_std - np.mean(channel_std)) / np.std(channel_std))
        mean_z = np.abs((channel_mean - np.mean(channel_mean)) / np.std(channel_mean))
        bad_channels_idx = np.where(
            (std_z > bad_channel_threshold) | (mean_z > bad_channel_threshold)
        )[0]
        bad_channels = [raw.ch_names[eeg_picks[i]] for i in bad_channels_idx]
        if bad_channels:
            print(f"    Found {len(bad_channels)} bad channel(s): {bad_channels}")
            raw.info["bads"].extend(bad_channels)
        else:
            print(f"    No bad channels detected")
    if interpolate_bads and len(raw.info["bads"]) > 0:
        print(f"  - Interpolating {len(raw.info['bads'])} bad channel(s)")
        bad_chan = raw.info["bads"]
        raw.interpolate_bads(reset_bads=True)
    else:
        bad_chan = "None"
    if "FCz" not in raw.ch_names:
        print("  - Adding FCz (online reference) back to channels")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Location for this channel is unknown"
            )
            raw.add_reference_channels("FCz")
        fc1_idx = raw.ch_names.index("FC1")
        fc2_idx = raw.ch_names.index("FC2")
        fc1_loc = raw.info["chs"][fc1_idx]["loc"]
        fc2_loc = raw.info["chs"][fc2_idx]["loc"]
        fc1_pos = fc1_loc[:3]
        fc2_pos = fc2_loc[:3]
        fcz_pos = (fc1_pos + fc2_pos) / 2
        fcz_idx = raw.ch_names.index("FCz")
        raw.info["chs"][fcz_idx]["loc"][:3] = fcz_pos
        print(
            f"  - FCz location set to: X={fcz_pos[0]}, Y={fcz_pos[1]}, Z={fcz_pos[2]}"
        )
    else:
        print("  - FCz already present in channels")
    print(f"  - Re-referencing to: {reference}")
    raw.set_eeg_reference("average", projection=False)
    if config.INTERACTIVE_MODE:
        raw.plot_sensors(show_names=True, show=True)
        plt.pause(0.1)
    ICA_len = 0
    ica = None
    eog_evoked = None
    print(
        f"  - Running ICA (method: {ica_method}, n_components: {ica_n_components or 'auto'})"
    )
    raw_ica = raw.copy()
    if l_freq < 1.0:
        print(f"    Applying additional 1 Hz high-pass filter for ICA")
        raw_ica.filter(1.0, None, fir_design="firwin")
    print(f"    Identifying flat/empty segments in data...")
    flat_annot = identify_flat_segments(
        raw_ica,
        win_size=2.0,
        step_size=1.0,
        variance_threshold=1e-12,
        min_active_channels=0.5,
        peak_to_peak_min=1e-06,
    )
    if flat_annot is not None:
        n_flat_segments = len(flat_annot)
        total_flat_duration = np.sum(flat_annot.duration)
        total_duration = raw_ica.times[-1]
        flat_percentage = total_flat_duration / total_duration * 100
        print(
            f"    Found {n_flat_segments} flat segment(s), total: {total_flat_duration:.1f}s ({flat_percentage:.1f}%)"
        )
        if raw_ica.annotations is None:
            raw_ica.set_annotations(flat_annot)
        else:
            raw_ica.set_annotations(raw_ica.annotations + flat_annot)
    else:
        print(f"    No flat segments detected")
    methods_to_try = [ica_method]
    if ica_method == "picard":
        methods_to_try.append("fastica")
    elif ica_method == "fastica":
        methods_to_try.append("picard")
    for method in methods_to_try:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ica = ICA(
                    n_components=ica_n_components,
                    method=method,
                    random_state=97,
                    max_iter=1000,
                )
                ica.fit(raw_ica, picks="eeg")
                converged = not any(
                    ("did not converge" in str(warn.message).lower() for warn in w)
                )
                if converged:
                    print(
                        f"    ICA fitted with {ica.n_components_} components using {method}"
                    )
                    break
                else:
                    print(
                        f"    Warning: {method} did not converge, trying next method..."
                    )
        except Exception as e:
            print(f"    ICA method {method} failed: {e}")
            continue
    print(f"    Detecting EOG/ECG artifacts (threshold-zscore:{ica_threshold[0]})...")
    ica.exclude = []
    eog_indices, eog_scores = ica.find_bads_eog(
        raw_ica, threshold=ica_threshold[0], measure="zscore", ch_name="VOGbelow"
    )
    ica.exclude = list(set(eog_indices))
    if eog_indices:
        print(
            f"    Found {len(eog_indices)} EOG component(s): {eog_indices}, with threshold: {ica_threshold[0]}"
        )
        if config.INTERACTIVE_MODE:
            print(
                f"    Inspecting detected components. (threshold-zscore:{ica_threshold[0]})"
            )
            plt.close('all')
            for idx in ica.exclude:
                ica.plot_properties(raw_ica, picks=idx, show=True)
        raw = ica.apply(raw)
        print(f"    ICA applied, {len(eog_indices)} component(s) removed")
        ICA_len = len(eog_indices)
        try:
            eog_evoked = mne.preprocessing.create_eog_epochs(raw_ica).average()
            eog_evoked.apply_baseline(baseline=(None, -0.2))
        except Exception as e:
            print(f"    Warning: Could not create EOG epochs for ICA: {e}")
    else:
        eog_indices_alt, _ = ica.find_bads_eog(
            raw_ica,
            threshold=ica_threshold[1],
            measure="correlation",
            ch_name="VOGbelow",
        )
        if eog_indices_alt:
            print(
                f"    Alternative detection found {len(eog_indices_alt)} component(s), with threshold-cor: {ica_threshold[1]}"
            )
            ica.exclude = list(set(eog_indices_alt))
            if config.INTERACTIVE_MODE:
                print(
                    f"    Inspecting detected components (threshold-correlation: {ica_threshold[1]})."
                )
                plt.close('all')
                for idx in ica.exclude:
                    ica.plot_properties(raw_ica, picks=idx, show=True)
            raw = ica.apply(raw)
            ICA_len = len(eog_indices_alt)
            try:
                eog_evoked = mne.preprocessing.create_eog_epochs(raw_ica).average()
                eog_evoked.apply_baseline(baseline=(None, -0.2))
            except Exception as e:
                print(f"    Warning: Could not create EOG epochs for ICA: {e}")
        elif manual_ica_detection:
            print(
                f"    No automatic detection, opening ICA component plots for manual inspection..."
            )
            if config.INTERACTIVE_MODE:
                plt.close('all')
                ica.plot_components(show=True)
            ica.plot_sources(raw_ica)
            manual_input = input(
                "    Enter component indices to exclude (comma-separated, e.g., '0,2,5') or press Enter to skip: "
            )
            if manual_input.strip():
                manual_indices = [int(x.strip()) for x in manual_input.split(",")]
                ica.exclude = manual_indices
                raw = ica.apply(raw)
                ICA_len = len(manual_indices)
                print(f"    Manually removed {ICA_len} component(s): {manual_indices}")
        else:
            print(f"    No EOG components detected")
    print("  Preprocessing complete!")
    return (raw, bad_chan, ICA_len, ica, eog_evoked)


def load_and_preprocess_subject_bids(
    bids_root: Path,
    subject: str,
    task: str,
    pipeline_name: str = "preprocessing",
    **preprocess_kwargs,
) -> bool:
    """Load, preprocess, and save EEG data for a single subject using BIDS + MNE-BIDS.

    Parameters:
        bids_root: Root of the BIDS dataset.
        subject: Subject ID (without 'sub-', e.g. '01').
        task: Task name (e.g. 'Flanker').
        pipeline_name: Name of the derivatives pipeline (default: 'preprocessing').
        **preprocess_kwargs: Passed directly to preprocess_eeg().

    Returns:
        bool: True if successful, False otherwise."""
    print(f"\n{'=' * 60}")
    print(f"Processing subject: sub-{subject}")
    print(f"{'=' * 60}")
    try:
        bids_path = BIDSPath(subject=subject, task=task, datatype="eeg", root=bids_root)
        print(f"Loading raw data via BIDS from: {bids_path.fpath}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(bids_path.fpath, preload=True)
        if "VOGbelow" in raw.ch_names:
            raw.set_channel_types({"VOGbelow": "eog"})
        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.2f} seconds")
        if len(raw.annotations) > 0:
            print(f"  Annotations/Events: {len(raw.annotations)}")
            unique_annotations = set(raw.annotations.description)
            print(f"  Unique event types: {len(unique_annotations)}")
        derivatives_root = bids_root / "derivatives" / pipeline_name
        derivatives_root.mkdir(parents=True, exist_ok=True)
        if not (derivatives_root / "dataset_description.json").exists():
            make_dataset_description(
                path=derivatives_root,
                name="EEG preprocessing pipeline",
                dataset_type="derivative",
                authors=["Anoushiravan Zahedi"],
                generated_by=[
                    {
                        "Name": pipeline_name,
                        "Description": "ICA-based artifact removal and preprocessing",
                    }
                ],
                source_datasets=[{"URL": str(bids_root)}],
            )
        print("\nRunning preprocessing pipeline...")
        raw_clean, bad_chan, len_ICA, ica_obj, eog_evoked = preprocess_eeg(
            raw, **preprocess_kwargs
        )
        deriv_bids_path = bids_path.copy().update(
            root=derivatives_root,
            processing="clean",
            suffix="eeg",
            extension=".fif",
            check=False,
        )
        deriv_bids_path.fpath.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n  Saving cleaned data to: {deriv_bids_path.fpath}")
        raw_clean.save(deriv_bids_path.fpath, overwrite=True)
        if ica_obj is not None:
            ica_path = deriv_bids_path.copy().update(
                suffix="ica", extension=".fif", check=False
            )
            ica_obj.save(ica_path.fpath, overwrite=True)
            print(f"  ICA object saved to: {ica_path.fpath}")
        if ica_obj is not None and len_ICA > 0:
            print(f"  Generating HTML report...")
            report = Report(title=f"ICA Report - sub-{subject}")
            report.add_ica(
                ica_obj,
                title="ICA Components",
                inst=raw_clean,
                eog_evoked=eog_evoked,
                picks=ica_obj.exclude,
            )
            report_path = (
                deriv_bids_path.copy()
                .update(suffix="report", extension=".html", check=False)
                .fpath
            )
            report.save(report_path, overwrite=True, open_browser=False)
            print(f"  HTML report saved to: {report_path}")
        sidecar_json = (
            deriv_bids_path.copy().update(extension=".json", check=False).fpath
        )
        metadata = {
            "Description": "EEG data after ICA-based artifact removal",
            "TaskName": task,
            "PreprocessingPipeline": pipeline_name,
            "PreprocessingParameters": preprocess_kwargs,
            "BadChannelsRemaining": bad_chan if bad_chan else [],
            "ICAComponentsRemoved": (
                [int(x) for x in ica_obj.exclude] if ica_obj else []
            ),
            "NumberICAComponentsRemoved": int(len_ICA),
            "Channels": int(len(raw_clean.ch_names)),
            "SamplingFrequency": float(raw_clean.info["sfreq"]),
            "DurationSeconds": float(raw_clean.times[-1]),
            "Software": "MNE-Python",
            "MNEVersion": mne.__version__,
        }
        with open(sidecar_json, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"  Metadata sidecar saved to: {sidecar_json}")
        log_path = (
            deriv_bids_path.copy()
            .update(suffix="log", extension=".txt", check=False)
            .fpath
        )
        with open(log_path, "w") as f:
            f.write(f"Subject: sub-{subject}\n")
            f.write(f"Task: {task}\n")
            f.write(f"Original BIDS path: {bids_path.fpath}\n")
            f.write(f"\nPreprocessing parameters:\n")
            for key, value in preprocess_kwargs.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nFinal info:\n")
            f.write(f"  Channels: {len(raw_clean.ch_names)}\n")
            f.write(f"  Sampling rate: {raw_clean.info['sfreq']} Hz\n")
            f.write(f"  Duration: {raw_clean.times[-1]:.2f} seconds\n")
            f.write(f"  Bad channels (remaining): {bad_chan}\n")
            f.write(f"  ICA removed components: {len_ICA}\n")
            if len_ICA > 0:
                f.write(f"  ICA component indices removed: {ica_obj.exclude}\n")
        print(f"  Preprocessing log saved to: {log_path}")
        print(f"\n✔ Successfully processed sub-{subject}")
        return True
    except Exception as e:
        print(f"\n✖ ERROR processing sub-{subject}: {e}")
        traceback.print_exc()
        return False
