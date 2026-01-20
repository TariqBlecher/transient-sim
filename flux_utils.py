"""Flux computation utilities for radio transient simulations.

Pure functions for spectral correction and expected flux calculations.
No dependencies beyond numpy and scipy.
"""

from typing import Optional

import numpy as np
from scipy import integrate


def compute_spectral_correction(spectral_index, reference_freq, freq_min, freq_max):
    """
    Compute the flux correction factor for spectral index.

    pfb applies: fprofile(v) = peak_flux * (v/v_ref)^alpha
    The cube averages over frequency, so effective flux = mean(fprofile) over band.

    Args:
        spectral_index: Power-law spectral index (alpha)
        reference_freq: Reference frequency in Hz
        freq_min: Minimum frequency of band in Hz
        freq_max: Maximum frequency of band in Hz

    Returns:
        Ratio of band-averaged flux to peak_flux
    """
    alpha = spectral_index

    if abs(alpha) < 1e-10:
        # Flat spectrum
        return 1.0

    if abs(alpha + 1) < 1e-10:
        # Special case: alpha = -1 (logarithmic integral)
        return reference_freq * np.log(freq_max / freq_min) / (freq_max - freq_min)

    # General case: integrate power law
    # integral of (v/v_ref)^alpha dv = v_ref^(-alpha) * v^(alpha+1) / (alpha+1)
    integral = (freq_max ** (alpha + 1) - freq_min ** (alpha + 1)) / (alpha + 1)
    average = integral / (freq_max - freq_min)

    return average / (reference_freq**alpha)


def compute_expected_flux(
    peak_flux,
    peak_time,
    duration,
    shape,
    bin_start,
    bin_end,
    spectral_index=0.0,
    reference_freq=None,
    freq_min=None,
    freq_max=None,
):
    """
    Compute expected measured flux in a time bin, accounting for shape and spectral index.

    Matches pfb-imaging implementation in pfb/utils/transients.py:
    - gaussian: sigma = duration (NOT FWHM-based)
    - exponential: tau = duration
    - step: active for [peak_time, peak_time + duration]
    - spectral index: flux(v) = peak_flux * (v/v_ref)^alpha, averaged over band

    Args:
        peak_flux: Peak flux in Jy
        peak_time: Time of transient peak in seconds
        duration: Duration parameter in seconds (interpretation depends on shape)
        shape: One of 'gaussian', 'exponential', 'step'
        bin_start: Start of time bin in seconds
        bin_end: End of time bin in seconds
        spectral_index: Power-law spectral index (default 0.0 = flat)
        reference_freq: Reference frequency in Hz (optional)
        freq_min: Minimum frequency in Hz (optional)
        freq_max: Maximum frequency in Hz (optional)

    Returns:
        Expected flux in the time bin in Jy
    """
    bin_width = bin_end - bin_start

    # Apply spectral correction if frequency info provided
    if reference_freq is not None and freq_min is not None and freq_max is not None:
        spec_correction = compute_spectral_correction(
            spectral_index, reference_freq, freq_min, freq_max
        )
    else:
        spec_correction = 1.0

    effective_peak_flux = peak_flux * spec_correction

    if shape == "step":
        # Step: constant flux from peak_time to peak_time + duration
        t_start = max(bin_start, peak_time)
        t_end = min(bin_end, peak_time + duration)
        overlap = max(0, t_end - t_start)
        return effective_peak_flux * overlap / bin_width

    elif shape == "gaussian":
        # Gaussian: sigma = duration (pfb convention)
        sigma = duration

        def gaussian(t):
            return effective_peak_flux * np.exp(
                -((t - peak_time) ** 2) / (2 * sigma**2)
            )

        integral, _ = integrate.quad(gaussian, bin_start, bin_end)
        return integral / bin_width

    elif shape == "exponential":
        # Exponential: tau = duration, starts at peak_time
        tau = duration
        if bin_end <= peak_time:
            return 0.0

        t_start = max(bin_start, peak_time)

        def exponential(t):
            return effective_peak_flux * np.exp(-(t - peak_time) / tau)

        integral, _ = integrate.quad(exponential, t_start, bin_end)
        return integral / bin_width

    return 0.0


def scale_peak_flux_for_snr(
    peak_flux: float,
    peak_time: float,
    duration: float,
    shape: str,
    integration_time: float,
    rms: float,
    snr_min: float,
    snr_max: float,
    rng: np.random.Generator,
    spectral_index: float = 0.0,
    reference_freq: Optional[float] = None,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
) -> tuple:
    """
    Scale peak flux to achieve target SNR range.

    Computes expected cube flux and scales peak_flux if SNR falls
    outside the target range.

    Args:
        peak_flux: Current peak flux in Jy
        peak_time: Time of transient peak in seconds
        duration: Duration parameter in seconds
        shape: Transient shape ('gaussian', 'exponential', 'step')
        integration_time: Integration time of observations in seconds
        rms: RMS noise level in Jy
        snr_min: Minimum target SNR
        snr_max: Maximum target SNR
        rng: Random number generator
        spectral_index: Power-law spectral index
        reference_freq: Reference frequency in Hz
        freq_min: Minimum frequency in Hz
        freq_max: Maximum frequency in Hz

    Returns:
        Tuple of (new_peak_flux, expected_cube_flux, was_scaled)
    """
    bin_start = peak_time - integration_time / 2
    bin_end = peak_time + integration_time / 2

    expected_flux = compute_expected_flux(
        peak_flux,
        peak_time,
        duration,
        shape,
        bin_start,
        bin_end,
        spectral_index=spectral_index,
        reference_freq=reference_freq,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    snr = expected_flux / rms

    if snr < snr_min or snr > snr_max:
        target_snr = float(rng.uniform(snr_min, snr_max))
        target_expected = target_snr * rms

        # Correction factor: expected_flux per unit peak_flux
        correction = compute_expected_flux(
            1.0,
            peak_time,
            duration,
            shape,
            bin_start,
            bin_end,
            spectral_index=spectral_index,
            reference_freq=reference_freq,
            freq_min=freq_min,
            freq_max=freq_max,
        )

        new_peak_flux = (
            target_expected / correction if correction > 0 else target_expected
        )
        return float(new_peak_flux), float(target_expected), True

    return float(peak_flux), float(expected_flux), False
