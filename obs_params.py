"""Observation parameter extraction from Measurement Sets.

Provides ObservationParams dataclass with factory method to extract
parameters from MS files using python-casacore.
"""

import numpy as np
from dataclasses import dataclass


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

    @classmethod
    def from_ms(cls, ms_path: str) -> 'ObservationParams':
        """Extract observation parameters from a Measurement Set.

        Args:
            ms_path: Path to the Measurement Set directory

        Returns:
            ObservationParams populated from the MS
        """
        from casacore.tables import table

        ms_path = str(ms_path)

        # Get time information
        with table(ms_path, readonly=True, ack=False) as t:
            times = t.getcol('TIME')
            unique_times = np.unique(times)

        time_range = times.max() - times.min()
        n_timesteps = len(unique_times)
        integration_time = float(np.median(np.diff(unique_times))) if n_timesteps > 1 else time_range

        # Get phase center
        with table(f'{ms_path}/FIELD', readonly=True, ack=False) as field_table:
            phase_dir = field_table.getcol('PHASE_DIR')[0][0]
            ra_deg = float(np.degrees(phase_dir[0])) % 360.0
            dec_deg = float(np.degrees(phase_dir[1]))

        # Get frequency information
        with table(f'{ms_path}/SPECTRAL_WINDOW', readonly=True, ack=False) as spw:
            ref_freq = float(spw.getcol('REF_FREQUENCY')[0])
            freqs = spw.getcol('CHAN_FREQ')[0]
            freq_min = float(freqs.min())
            freq_max = float(freqs.max())

        return cls(
            ms_path=ms_path,
            time_range_sec=float(time_range),
            n_timesteps=n_timesteps,
            integration_time_sec=integration_time,
            ra_center_deg=ra_deg,
            dec_center_deg=dec_deg,
            reference_freq_hz=ref_freq,
            freq_min_hz=freq_min,
            freq_max_hz=freq_max
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
    with table(f'{ms_path}/SPECTRAL_WINDOW', readonly=True, ack=False) as spw:
        ref_freq = float(spw.getcol('REF_FREQUENCY')[0])
        freqs = spw.getcol('CHAN_FREQ')[0]
        freq_min = float(freqs.min())
        freq_max = float(freqs.max())

    return freq_min, freq_max, ref_freq
