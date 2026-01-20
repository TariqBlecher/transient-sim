"""Observation parameter extraction from Measurement Sets.

Provides ObservationParams dataclass with factory method to extract
parameters from MS files using python-casacore.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ScanInfo:
    """Information about a single scan in the observation."""

    scan_id: int
    field_id: int
    field_name: str
    start_time_sec: float  # Relative to observation start
    end_time_sec: float  # Relative to observation start

    @property
    def duration_sec(self) -> float:
        """Duration of this scan in seconds."""
        return self.end_time_sec - self.start_time_sec


@dataclass
class ObservationParams:
    """Parameters extracted from a Measurement Set."""

    ms_path: str
    time_range_sec: float
    n_timesteps: int
    integration_time_sec: float
    ra_center_deg: float
    dec_center_deg: float
    reference_freq_hz: float
    freq_min_hz: float
    freq_max_hz: float
    # Optional scan information (populated when extract_scans=True)
    scans: Optional[List[ScanInfo]] = field(default=None)
    target_field_id: Optional[int] = field(default=None)

    def get_science_scans(self) -> List[ScanInfo]:
        """Return scans belonging to the target (science) field.

        Returns:
            List of ScanInfo for scans with field_id == target_field_id.
            Empty list if no scan info available.
        """
        if self.scans is None or self.target_field_id is None:
            return []
        return [s for s in self.scans if s.field_id == self.target_field_id]

    def sample_science_time(self, rng: np.random.Generator) -> float:
        """Sample a random time from within science scans, weighted by duration.

        Args:
            rng: NumPy random generator

        Returns:
            Random time (seconds from observation start) within a science scan.

        Raises:
            ValueError: If no science scans are available.
        """
        science_scans = self.get_science_scans()
        if not science_scans:
            raise ValueError("No science scans available for sampling")

        # Weight by scan duration for uniform sampling across total science time
        durations = np.array([s.duration_sec for s in science_scans])
        total_duration = durations.sum()
        weights = durations / total_duration

        # Select a scan proportional to its duration
        scan_idx = rng.choice(len(science_scans), p=weights)
        selected_scan = science_scans[scan_idx]

        # Sample uniformly within the selected scan
        return float(rng.uniform(selected_scan.start_time_sec, selected_scan.end_time_sec))

    @classmethod
    def from_ms(cls, ms_path: str, extract_scans: bool = True) -> "ObservationParams":
        """Extract observation parameters from a Measurement Set.

        Args:
            ms_path: Path to the Measurement Set directory
            extract_scans: If True, extract scan boundaries to enable scan-aware
                peak time sampling. Set to False for backward compatibility.

        Returns:
            ObservationParams populated from the MS
        """
        from casacore.tables import table

        ms_path = str(ms_path)

        # Get time information
        with table(ms_path, readonly=True, ack=False) as t:
            times = t.getcol("TIME")
            unique_times = np.unique(times)

        time_range = times.max() - times.min()
        n_timesteps = len(unique_times)
        integration_time = (
            float(np.median(np.diff(unique_times))) if n_timesteps > 1 else time_range
        )

        # Get phase center
        with table(f"{ms_path}/FIELD", readonly=True, ack=False) as field_table:
            phase_dir = field_table.getcol("PHASE_DIR")[0][0]
            ra_deg = float(np.degrees(phase_dir[0])) % 360.0
            dec_deg = float(np.degrees(phase_dir[1]))

        # Get frequency information
        with table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True, ack=False) as spw:
            ref_freq = float(spw.getcol("REF_FREQUENCY")[0])
            freqs = spw.getcol("CHAN_FREQ")[0]
            freq_min = float(freqs.min())
            freq_max = float(freqs.max())

        # Extract scan info if requested
        scans = None
        target_field_id = None
        if extract_scans:
            scans, target_field_id = get_scan_info_from_ms(ms_path)

        return cls(
            ms_path=ms_path,
            time_range_sec=float(time_range),
            n_timesteps=n_timesteps,
            integration_time_sec=integration_time,
            ra_center_deg=ra_deg,
            dec_center_deg=dec_deg,
            reference_freq_hz=ref_freq,
            freq_min_hz=freq_min,
            freq_max_hz=freq_max,
            scans=scans,
            target_field_id=target_field_id,
        )


def get_frequency_range_from_ms(ms_path):
    """Get frequency range from a Measurement Set.

    Convenience function for cases where only frequency info is needed.

    Args:
        ms_path: Path to the Measurement Set directory

    Returns:
        Tuple of (freq_min_hz, freq_max_hz, reference_freq_hz)
    """
    from casacore.tables import table

    ms_path = str(ms_path)
    with table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True, ack=False) as spw:
        ref_freq = float(spw.getcol("REF_FREQUENCY")[0])
        freqs = spw.getcol("CHAN_FREQ")[0]
        freq_min = float(freqs.min())
        freq_max = float(freqs.max())

    return freq_min, freq_max, ref_freq


def get_scan_info_from_ms(ms_path: str) -> Tuple[List[ScanInfo], int]:
    """Extract scan information from a Measurement Set.

    Reads TIME, SCAN_NUMBER, and FIELD_ID columns to identify scans,
    then maps field IDs to names from the FIELD subtable.

    Args:
        ms_path: Path to the Measurement Set directory

    Returns:
        Tuple of (list of ScanInfo objects, target_field_id)
        Target field is field 0 (same as phase center extraction).
    """
    from casacore.tables import table

    ms_path = str(ms_path)

    # Get field names from FIELD subtable
    with table(f"{ms_path}/FIELD", readonly=True, ack=False) as field_table:
        field_names = list(field_table.getcol("NAME"))

    # Read main table columns
    with table(ms_path, readonly=True, ack=False) as t:
        times = t.getcol("TIME")
        scan_numbers = t.getcol("SCAN_NUMBER")
        field_ids = t.getcol("FIELD_ID")

    # Get observation start time for relative timestamps
    obs_start = times.min()

    # Build scan info by grouping rows
    # Each unique (scan_number, field_id) combination is a scan
    scan_data = {}
    for i in range(len(times)):
        scan_id = int(scan_numbers[i])
        field_id = int(field_ids[i])
        time_val = times[i]
        key = (scan_id, field_id)

        if key not in scan_data:
            scan_data[key] = {"times": [], "field_id": field_id, "scan_id": scan_id}
        scan_data[key]["times"].append(time_val)

    # Convert to ScanInfo objects
    scans = []
    for (scan_id, field_id), data in scan_data.items():
        times_arr = np.array(data["times"])
        field_name = field_names[field_id] if field_id < len(field_names) else f"field_{field_id}"
        scans.append(
            ScanInfo(
                scan_id=scan_id,
                field_id=field_id,
                field_name=field_name,
                start_time_sec=float(times_arr.min() - obs_start),
                end_time_sec=float(times_arr.max() - obs_start),
            )
        )

    # Sort by start time
    scans.sort(key=lambda s: s.start_time_sec)

    # Target field is field 0 (consistent with phase center extraction)
    target_field_id = 0

    return scans, target_field_id
