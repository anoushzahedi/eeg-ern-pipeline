from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, make_dataset_description
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import warnings


def extract_peak_amplitude(
    evoked: mne.Evoked,
    tmin: float = 0.0,
    tmax: float = 0.1,
    channel: str = "FCz",
    mode: str = "neg",
) -> Tuple[float, float]:
    """Extract peak amplitude and latency from an evoked response.

    Parameters
    ----------
    evoked : mne.Evoked
        Evoked response.
    tmin : float
        Start of time window (default: 0.0s).
    tmax : float
        End of time window (default: 0.1s = 100ms).
    channel : str
        Channel name (default: 'FCz').
    mode : str
        'neg' for most negative, 'pos' for most positive.

    Returns
    -------
    peak_amplitude : float
        Peak amplitude in µV.
    peak_latency : float
        Peak latency in seconds."""
    if channel not in evoked.ch_names:
        available_channels = [ch for ch in evoked.ch_names if "FC" in ch or ch == "Cz"]
        raise ValueError(
            f"Channel {channel} not found. Available frontocentral channels: {available_channels}"
        )
    ch_idx = evoked.ch_names.index(channel)
    time_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
    data_in_window = evoked.data[ch_idx, time_mask]
    times_in_window = evoked.times[time_mask]
    if mode == "neg":
        peak_idx = np.argmin(data_in_window)
    else:
        peak_idx = np.argmax(data_in_window)
    peak_amplitude = data_in_window[peak_idx] * 1000000.0
    peak_latency = times_in_window[peak_idx]
    return (peak_amplitude, peak_latency)


def extract_mean_amplitude(
    evoked: mne.Evoked, tmin: float = 0.0, tmax: float = 0.1, channel: str = "FCz"
) -> float:
    """Extract mean amplitude in a time window.

    Parameters
    ----------
    evoked : mne.Evoked
        Evoked response.
    tmin : float
        Start of time window.
    tmax : float
        End of time window.
    channel : str
        Channel name.

    Returns
    -------
    mean_amplitude : float
        Mean amplitude in µV."""
    if channel not in evoked.ch_names:
        available_channels = [ch for ch in evoked.ch_names if "FC" in ch or ch == "Cz"]
        raise ValueError(
            f"Channel {channel} not found. Available: {available_channels}"
        )
    ch_idx = evoked.ch_names.index(channel)
    time_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
    mean_amplitude = np.mean(evoked.data[ch_idx, time_mask]) * 1000000.0
    return mean_amplitude


def process_subject_ern(
    bids_root: Path,
    subject: str,
    task: str = "Flanker",
    epochs_pipeline: str = "epochs",
    ern_pipeline: str = "ern",
    channel: str = "FCz",
    ern_tmin: float = 0.0,
    ern_tmax: float = 0.1,
    use_peak: bool = True,
    save_plots: bool = True,
) -> Dict:
    """Process one subject: load epochs, compute ERN/CRN, extract amplitudes.

    This function processes a single subject independently and saves results
    to the ERN derivatives folder.

    Parameters
    ----------
    bids_root : Path
        Root directory of BIDS dataset.
    subject : str
        Subject ID (without 'sub-' prefix).
    task : str
        Task name (default: 'Flanker').
    epochs_pipeline : str
        Name of epochs derivative folder (default: 'epochs').
    ern_pipeline : str
        Name of ERN derivative folder (default: 'ern').
    channel : str
        Channel to analyze (default: 'FCz').
    ern_tmin : float
        Start of ERN time window in seconds (default: 0.0).
    ern_tmax : float
        End of ERN time window in seconds (default: 0.1).
    use_peak : bool
        If True, extract peak amplitude; if False, mean amplitude.
    save_plots : bool
        If True, save ERP plots and topographies.

    Returns
    -------
    results : dict
        Dictionary containing all ERN metrics for this subject."""
    print(f"\n{'=' * 60}")
    print(f"Processing subject: sub-{subject}")
    print(f"{'=' * 60}")
    epochs_root = bids_root / "derivatives" / epochs_pipeline
    ern_root = bids_root / "derivatives" / ern_pipeline
    if not (ern_root / "dataset_description.json").exists():
        make_dataset_description(
            path=ern_root,
            name="EEG ERN extraction pipeline",
            dataset_type="derivative",
            authors=["Anoushiravan Zahedi"],
            generated_by=[
                {
                    "Name": ern_pipeline,
                    "Description": "Extracted ERNs based on peak amplitude.",
                }
            ],
            source_datasets=[{"URL": str(epochs_root)}],
        )
    epochs_path = BIDSPath(
        subject=subject,
        task=task,
        datatype="eeg",
        suffix="epo",
        extension=".fif",
        root=epochs_root,
        check=False,
    )
    print(f"Loading epochs from: {epochs_path.fpath}")
    if not epochs_path.fpath.exists():
        raise FileNotFoundError(f"Epochs file not found: {epochs_path.fpath}")
    epochs = mne.read_epochs(epochs_path.fpath, preload=True)
    results = {
        "subject": subject,
        "channel": channel,
        "ern_window_start_ms": ern_tmin * 1000,
        "ern_window_end_ms": ern_tmax * 1000,
        "analysis_method": "peak" if use_peak else "mean",
        "n_epochs_total": len(epochs),
    }
    event_categories = {
        "Incorrect_0%": [
            "Response_Incorrect_slow_0%_Congruency",
            "Response_Incorrect_fast_0%_Congruency",
        ],
        "Correct_0%": [
            "Response_Correct_slow_0%_Congruency",
            "Response_Correct_fast_0%_Congruency",
        ],
        "Incorrect_33%": [
            "Response_Incorrect_slow_33%_Congruency",
            "Response_Incorrect_fast_33%_Congruency",
        ],
        "Correct_33%": [
            "Response_Correct_slow_33%_Congruency",
            "Response_Correct_fast_33%_Congruency",
        ],
        "Incorrect_66%": [
            "Response_Incorrect_slow_66%_Congruency",
            "Response_Incorrect_fast_66%_Congruency",
        ],
        "Correct_66%": [
            "Response_Correct_slow_66%_Congruency",
            "Response_Correct_fast_66%_Congruency",
        ],
        "Incorrect_100%": [
            "Response_Incorrect_slow_100%_Congruency",
            "Response_Incorrect_fast_100%_Congruency",
        ],
        "Correct_100%": [
            "Response_Correct_slow_100%_Congruency",
            "Response_Correct_fast_100%_Congruency",
        ],
        "Incorrect_all": ["Response_Incorrect"],
        "Correct_all": ["Response_Correct"],
    }
    evokeds = {}
    for category_name, event_patterns in event_categories.items():
        try:
            if category_name in ["Incorrect_all", "Correct_all"]:
                pattern = event_patterns[0]
                matching_events = [e for e in epochs.event_id.keys() if pattern in e]
                if matching_events:
                    category_epochs = epochs[matching_events]
                else:
                    print(f"  WARNING: No events found matching '{pattern}'")
                    continue
            else:
                matching_events = [e for e in event_patterns if e in epochs.event_id]
                if not matching_events:
                    print(
                        f"  WARNING: Events not found for {category_name}: {event_patterns}"
                    )
                    continue
                category_epochs = epochs[matching_events]
            n_epochs = len(category_epochs)
            results[f"n_epochs_{category_name}"] = n_epochs
            if n_epochs == 0:
                print(f"  WARNING: No epochs for {category_name}")
                continue
            evoked = category_epochs.average()
            evokeds[category_name] = evoked
            if use_peak:
                amplitude, latency = extract_peak_amplitude(
                    evoked, ern_tmin, ern_tmax, channel, mode="neg"
                )
                results[f"{category_name}_peak_amplitude_uV"] = amplitude
                results[f"{category_name}_peak_latency_ms"] = latency * 1000
                print(
                    f"  {category_name}: {n_epochs} epochs, peak = {amplitude:.2f} µV at {latency * 1000:.1f} ms"
                )
            else:
                amplitude = extract_mean_amplitude(evoked, ern_tmin, ern_tmax, channel)
                results[f"{category_name}_mean_amplitude_uV"] = amplitude
                print(
                    f"  {category_name}: {n_epochs} epochs, mean = {amplitude:.2f} µV"
                )
        except Exception as e:
            print(f"  ERROR processing {category_name}: {e}")
            continue
    print("\nCalculating difference waves:")
    if "Incorrect_0%" in evokeds and "Correct_0%" in evokeds:
        if use_peak:
            delta_ern_0 = (
                results["Incorrect_0%_peak_amplitude_uV"]
                - results["Correct_0%_peak_amplitude_uV"]
            )
        else:
            delta_ern_0 = (
                results["Incorrect_0%_mean_amplitude_uV"]
                - results["Correct_0%_mean_amplitude_uV"]
            )
        results["ΔERN_0%_congruency_uV"] = delta_ern_0
        print(f"  ΔERN (0% congruency) = {delta_ern_0:.2f} µV")
    
    if "Incorrect_33%" in evokeds and "Correct_33%" in evokeds:
        if use_peak:
            delta_ern_33 = (
                results["Incorrect_33%_peak_amplitude_uV"]
                - results["Correct_33%_peak_amplitude_uV"]
            )
        else:
            delta_ern_33 = (
                results["Incorrect_33%_mean_amplitude_uV"]
                - results["Correct_33%_mean_amplitude_uV"]
            )
        results["ΔERN_33%_congruency_uV"] = delta_ern_33
        print(f"  ΔERN (33% congruency) = {delta_ern_33:.2f} µV")
    
    if "Incorrect_66%" in evokeds and "Correct_66%" in evokeds:
        if use_peak:
            delta_ern_66 = (
                results["Incorrect_66%_peak_amplitude_uV"]
                - results["Correct_66%_peak_amplitude_uV"]
            )
        else:
            delta_ern_66 = (
                results["Incorrect_66%_mean_amplitude_uV"]
                - results["Correct_66%_mean_amplitude_uV"]
            )
        results["ΔERN_66%_congruency_uV"] = delta_ern_66
        print(f"  ΔERN (66% congruency) = {delta_ern_66:.2f} µV")
    
    if "Incorrect_100%" in evokeds and "Correct_100%" in evokeds:
        if use_peak:
            delta_ern_100 = (
                results["Incorrect_100%_peak_amplitude_uV"]
                - results["Correct_100%_peak_amplitude_uV"]
            )
        else:
            delta_ern_100 = (
                results["Incorrect_100%_mean_amplitude_uV"]
                - results["Correct_100%_mean_amplitude_uV"]
            )
        results["ΔERN_100%_congruency_uV"] = delta_ern_100
        print(f"  ΔERN (100% congruency) = {delta_ern_100:.2f} µV")
    
    if "Incorrect_all" in evokeds and "Correct_all" in evokeds:
        if use_peak:
            delta_ern_overall = (
                results["Incorrect_all_peak_amplitude_uV"]
                - results["Correct_all_peak_amplitude_uV"]
            )
        else:
            delta_ern_overall = (
                results["Incorrect_all_mean_amplitude_uV"]
                - results["Correct_all_mean_amplitude_uV"]
            )
        results["ΔERN_overall_uV"] = delta_ern_overall
        print(f"  ΔERN (overall) = {delta_ern_overall:.2f} µV")
    
    subject_output_dir = ern_root / f"sub-{subject}" / "eeg"
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = subject_output_dir / f"sub-{subject}_task-{task}_ern.csv"
    df_single = pd.DataFrame([results])
    df_single.to_csv(csv_path, index=False)
    print(f"\n✓ Individual CSV saved to: {csv_path}")
    
    # Save evoked objects for grand average computation
    print("\nSaving evoked objects:")
    for category_name, evoked in evokeds.items():
        evoked_path = subject_output_dir / f"sub-{subject}_task-{task}_condition-{category_name}-ave.fif"
        evoked.save(evoked_path, overwrite=True)
        print(f"  ✓ Saved: {evoked_path.name}")
    
    if save_plots and evokeds:
        _save_ern_plots(
            evokeds=evokeds,
            subject=subject,
            task=task,
            channel=channel,
            ern_tmin=ern_tmin,
            ern_tmax=ern_tmax,
            output_dir=subject_output_dir,
        )
    return results


def _save_ern_plots(
    evokeds: Dict[str, mne.Evoked],
    subject: str,
    task: str,
    channel: str,
    ern_tmin: float,
    ern_tmax: float,
    output_dir: Path,
) -> None:
    """Save ERP plots for a subject."""
    # Create 2x2 subplot grid for all four congruency levels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    congruency_levels = [
        ("0%", "0% Congruency (Incongruent)", 0),
        ("33%", "33% Congruency", 1),
        ("66%", "66% Congruency", 2),
        ("100%", "100% Congruency (Congruent)", 3),
    ]
    
    for cong_key, cong_title, ax_idx in congruency_levels:
        incorrect_key = f"Incorrect_{cong_key}"
        correct_key = f"Correct_{cong_key}"
        
        if incorrect_key in evokeds and correct_key in evokeds:
            ax = axes[ax_idx]
            times = evokeds[incorrect_key].times * 1000
            ch_idx = evokeds[incorrect_key].ch_names.index(channel)
            incorrect_data = evokeds[incorrect_key].data[ch_idx] * 1000000.0
            correct_data = evokeds[correct_key].data[ch_idx] * 1000000.0
            
            ax.plot(times, incorrect_data, "r-", linewidth=2, label="Incorrect")
            ax.plot(times, correct_data, "b-", linewidth=2, label="Correct")
            ax.axvline(0, color="k", linestyle="--", alpha=0.5)
            ax.axhline(0, color="k", linestyle="-", alpha=0.3)
            ax.axvspan(
                ern_tmin * 1000,
                ern_tmax * 1000,
                alpha=0.2,
                color="gray",
                label="ERN window",
            )
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (µV)")
            ax.set_title(f"sub-{subject}: {cong_title}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # If data not available, hide the subplot
            axes[ax_idx].axis('off')
            axes[ax_idx].text(
                0.5, 0.5, 
                f"No data for {cong_key} congruency",
                ha='center', va='center',
                transform=axes[ax_idx].transAxes
            )
    
    plt.tight_layout()
    plot_path = output_dir / f"sub-{subject}_task-{task}_ern.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved to: {plot_path}")
    
    # Save topography for overall incorrect responses
    if "Incorrect_all" in evokeds:
        fig = evokeds["Incorrect_all"].plot_topomap(
            times=[0.05, 0.08], average=0.02, show=False
        )
        # Update the figure title
        fig.suptitle(f"sub-{subject}: Incorrect Responses (All Congruency Levels)", 
                     fontsize=12, y=0.98)
        topo_path = output_dir / f"sub-{subject}_task-{task}_ern-topo.png"
        fig.savefig(topo_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Topography saved to: {topo_path}")


def merge_ern_results(
    bids_root: Path,
    task: str = "Flanker",
    ern_pipeline: str = "ern",
    output_filename: str = None,
    channel: str = "FCz",
    compute_grand_average: bool = True,
    save_grand_average_evoked: bool = True,
) -> pd.DataFrame:
    """Merge individual participant CSV files into a single summary CSV.

    This function should be run AFTER processing all subjects to combine
    their individual results into a group-level summary file.

    Parameters
    ----------
    bids_root : Path
        Root directory of BIDS dataset.
    task : str
        Task name (default: 'Flanker').
    ern_pipeline : str
        Name of ERN derivative folder (default: 'ern').
    output_filename : str, optional
        Custom filename for output CSV. If None, uses default naming.
    channel : str
        Channel used in analysis (for filename).
    compute_grand_average : bool
        If True, compute and plot grand averages across all subjects (default: True).
    save_grand_average_evoked : bool
        If True, save grand average evoked objects as .fif files (default: True).

    Returns
    -------
    df_merged : pd.DataFrame
        Merged dataframe with all subjects' results."""
    print(f"\n{'=' * 60}")
    print("MERGING INDIVIDUAL ERN RESULTS")
    print(f"{'=' * 60}\n")
    ern_root = bids_root / "derivatives" / ern_pipeline
    csv_files = sorted(ern_root.glob(f"sub-*/eeg/sub-*_task-{task}_ern.csv"))
    print(f"Found {len(csv_files)} individual ERN CSV files")
    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No individual CSV files found in {ern_root}/sub-*/eeg/\nMake sure you have run process_subject_ern() for all subjects first."
        )
    dfs = []
    for csv_file in csv_files:
        df_single = pd.read_csv(csv_file)
        dfs.append(df_single)
        print(f"  ✓ Loaded: {csv_file.name}")
    df_merged = pd.concat(dfs, ignore_index=True)
    if output_filename is None:
        output_filename = f"ern_amplitudes_{channel}_summary.csv"
    output_csv = ern_root / output_filename
    df_merged.to_csv(output_csv, index=False)
    print(f"\n{'=' * 60}")
    print(f"✓ Merged CSV saved to: {output_csv}")
    print(f"{'=' * 60}")
    print(f"Total subjects: {len(df_merged)}")
    _print_summary_statistics(df_merged, channel)
    
    # Compute and plot grand averages
    if compute_grand_average:
        print(f"\n{'=' * 60}")
        print("COMPUTING GRAND AVERAGES")
        print(f"{'=' * 60}\n")
        _compute_and_plot_grand_averages(
            ern_root=ern_root,
            task=task,
            channel=channel,
            save_evoked=save_grand_average_evoked,
        )
    
    return df_merged


def _compute_and_plot_grand_averages(
    ern_root: Path,
    task: str,
    channel: str,
    save_evoked: bool = True,
) -> None:
    """Compute grand averages across all subjects and create plots.
    
    Parameters
    ----------
    ern_root : Path
        Root directory of ERN derivatives.
    task : str
        Task name.
    channel : str
        Channel to plot.
    save_evoked : bool
        If True, save grand average evoked objects."""
    
    # Define all conditions to process
    conditions = [
        "Incorrect_0%", "Correct_0%",
        "Incorrect_33%", "Correct_33%",
        "Incorrect_66%", "Correct_66%",
        "Incorrect_100%", "Correct_100%",
        "Incorrect_all", "Correct_all",
    ]
    
    grand_averages = {}
    
    # Load and average evoked data for each condition
    for condition in conditions:
        evoked_files = sorted(ern_root.glob(f"sub-*/eeg/sub-*_task-{task}_condition-{condition}-ave.fif"))
        
        if len(evoked_files) == 0:
            print(f"  WARNING: No evoked files found for {condition}")
            continue
        
        evokeds_list = []
        for evoked_file in evoked_files:
            try:
                evoked = mne.read_evokeds(evoked_file)[0]
                evokeds_list.append(evoked)
            except Exception as e:
                print(f"  WARNING: Could not load {evoked_file.name}: {e}")
                continue
        
        if len(evokeds_list) > 0:
            grand_avg = mne.grand_average(evokeds_list)
            grand_averages[condition] = grand_avg
            print(f"  ✓ Grand average for {condition}: {len(evokeds_list)} subjects")
            del evokeds_list
            
            # Save grand average evoked object
            if save_evoked:
                ga_path = ern_root / f"grand_average_task-{task}_condition-{condition}-ave.fif"
                grand_avg.save(ga_path, overwrite=True)
                print(f"    Saved to: {ga_path.name}")
    
    # Plot grand averages
    if grand_averages:
        print("\nCreating grand average plots...")
        _plot_grand_average_ern(
            grand_averages=grand_averages,
            task=task,
            channel=channel,
            output_dir=ern_root,
        )


def _plot_grand_average_ern(
    grand_averages: Dict[str, mne.Evoked],
    task: str,
    channel: str,
    output_dir: Path,
) -> None:
    """Plot grand average ERPs across all subjects.
    
    Parameters
    ----------
    grand_averages : dict
        Dictionary of grand average evoked objects.
    task : str
        Task name.
    channel : str
        Channel to plot.
    output_dir : Path
        Directory to save plots."""
    
    # Create 2x2 subplot grid for all four congruency levels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    congruency_levels = [
        ("0%", "0% Congruency (Incongruent)", 0),
        ("33%", "33% Congruency", 1),
        ("66%", "66% Congruency", 2),
        ("100%", "100% Congruency (Congruent)", 3),
    ]
    
    for cong_key, cong_title, ax_idx in congruency_levels:
        incorrect_key = f"Incorrect_{cong_key}"
        correct_key = f"Correct_{cong_key}"
        
        if incorrect_key in grand_averages and correct_key in grand_averages:
            ax = axes[ax_idx]
            
            # Get data
            times = grand_averages[incorrect_key].times * 1000
            ch_idx = grand_averages[incorrect_key].ch_names.index(channel)
            incorrect_data = grand_averages[incorrect_key].data[ch_idx] * 1000000.0
            correct_data = grand_averages[correct_key].data[ch_idx] * 1000000.0
            
            # Plot lines
            ax.plot(times, incorrect_data, "r-", linewidth=2.5, label="Incorrect")
            ax.plot(times, correct_data, "b-", linewidth=2.5, label="Correct")
            
            # Add reference lines
            ax.axvline(0, color="k", linestyle="--", alpha=0.5)
            ax.axhline(0, color="k", linestyle="-", alpha=0.3)
            
            # Styling
            ax.set_xlabel("Time (ms)", fontsize=11)
            ax.set_ylabel("Amplitude (µV)", fontsize=11)
            ax.set_title(f"Grand Average (N={len(grand_averages)}): {cong_title}", fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            # If data not available, hide the subplot
            axes[ax_idx].axis('off')
            axes[ax_idx].text(
                0.5, 0.5, 
                f"No data for {cong_key} congruency",
                ha='center', va='center',
                transform=axes[ax_idx].transAxes,
                fontsize=12
            )
    
    plt.tight_layout()
    plot_path = output_dir / f"grand_average_task-{task}_ern.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close('all')
    print(f"\n✓ Grand average plot saved to: {plot_path}")
    
    # Create grand average topography for overall incorrect
    if "Incorrect_all" in grand_averages:
        fig = grand_averages["Incorrect_all"].plot_topomap(
            times=[0.05, 0.08], average=0.02, show=False
        )
        fig.suptitle(f"Grand Average: Incorrect Responses (All Congruency Levels)", 
                     fontsize=12, y=0.98, fontweight='bold')
        topo_path = output_dir / f"grand_average_task-{task}_ern-topo.png"
        fig.savefig(topo_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Grand average topography saved to: {topo_path}")


def _print_summary_statistics(df: pd.DataFrame, channel: str) -> None:
    """Print summary statistics from merged dataframe."""
    print(f"\nSummary statistics:")
    print(f"  Channel: {channel}")
    if "ern_window_start_ms" in df.columns and "ern_window_end_ms" in df.columns:
        ern_start = df["ern_window_start_ms"].iloc[0]
        ern_end = df["ern_window_end_ms"].iloc[0]
        print(f"  ERN window: {ern_start:.0f}-{ern_end:.0f} ms")
    if "analysis_method" in df.columns:
        method = df["analysis_method"].iloc[0]
        print(f"  Method: {method.capitalize()} amplitude")
    if "ΔERN_overall_uV" in df.columns:
        print(f"\n  ΔERN (overall):")
        print(f"    Mean: {df['ΔERN_overall_uV'].mean():.2f} µV")
        print(f"    SD: {df['ΔERN_overall_uV'].std():.2f} µV")
        print(
            f"    Range: [{df['ΔERN_overall_uV'].min():.2f}, {df['ΔERN_overall_uV'].max():.2f}] µV"
        )
    
    # Print statistics for all congruency levels
    congruency_columns = [
        ("ΔERN_0%_congruency_uV", "0% (Incongruent)"),
        ("ΔERN_33%_congruency_uV", "33%"),
        ("ΔERN_66%_congruency_uV", "66%"),
        ("ΔERN_100%_congruency_uV", "100% (Congruent)"),
    ]
    
    available_congruencies = [
        (col, label) for col, label in congruency_columns if col in df.columns
    ]
    
    if available_congruencies:
        print(f"\n  ΔERN by congruency:")
        for col, label in available_congruencies:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"    {label}: {mean_val:.2f} ± {std_val:.2f} µV")