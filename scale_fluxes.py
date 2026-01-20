"""Scale transient fluxes to achieve target SNR range in output cubes."""

import argparse
import numpy as np

from manifest_io import load_manifest_ecsv, save_manifest_ecsv
from flux_utils import compute_expected_flux
from cube_utils import extract_rms_from_cube
from obs_params import ObservationParams
from transient import Transient, save_transients_yaml


def scale_fluxes(manifest_path, ms_path, snr_min=5.0, snr_max=20.0, seed=None,
                 rms=None, rms_cube_path=None):
    """
    Scale transient fluxes to ensure sensible, detectable values.

    This function ensures transients have realistic fluxes for detection testing.
    Without scaling, randomly-generated fluxes may produce transients that are
    either too faint to detect or unrealistically bright compared to the noise.

    Typically called as part of transient generation (via transient_sim.py), but
    available as standalone CLI for re-scaling existing manifests.

    For each transient:
    - Compute expected_cube_flux from current peak_flux
    - Calculate SNR = expected_cube_flux / rms
    - If SNR < snr_min or SNR > snr_max, scale peak_flux to hit random target in range

    Args:
        manifest_path: Path to input manifest ECSV
        ms_path: Path to Measurement Set
        snr_min: Minimum target SNR (default 5.0)
        snr_max: Maximum target SNR (default 20.0)
        seed: Random seed for reproducibility
        rms: RMS noise level in Jy (if provided, rms_cube_path is ignored)
        rms_cube_path: Path to baseline zarr cube for RMS measurement (used if rms not provided)

    Returns:
        Tuple of (list of Transient objects, metadata dict)
    """
    if rms is None and rms_cube_path is None:
        raise ValueError("Either rms or rms_cube_path must be provided")

    rng = np.random.default_rng(seed)

    # Load inputs
    df, metadata = load_manifest_ecsv(manifest_path)
    if rms is None:
        rms = extract_rms_from_cube(rms_cube_path)
    obs = ObservationParams.from_ms(ms_path)

    print(f"Loaded {len(df)} transients")
    print(f"RMS from cube: {rms:.4f} Jy")
    print(f"Target SNR range: {snr_min}-{snr_max}")

    scaled_count = 0
    transients = []

    for _, row in df.iterrows():
        # Convert row to dict for factory method
        row_dict = row.to_dict()
        peak_flux = row_dict['peak_flux_jy']
        peak_time = row_dict['peak_time_sec']
        duration = row_dict['duration_sec']
        shape = row_dict['shape']
        spectral_index = row_dict.get('spectral_index', 0.0)

        # Compute expected flux with current peak_flux
        bin_start = peak_time - obs.integration_time_sec / 2
        bin_end = peak_time + obs.integration_time_sec / 2

        expected_flux = compute_expected_flux(
            peak_flux, peak_time, duration, shape, bin_start, bin_end,
            spectral_index=spectral_index,
            reference_freq=obs.reference_freq_hz,
            freq_min=obs.freq_min_hz,
            freq_max=obs.freq_max_hz
        )

        snr = expected_flux / rms

        # Scale if outside range
        if snr < snr_min or snr > snr_max:
            target_snr = float(rng.uniform(snr_min, snr_max))
            target_expected = target_snr * rms

            # Correction factor: expected_flux per unit peak_flux
            correction = compute_expected_flux(
                1.0, peak_time, duration, shape, bin_start, bin_end,
                spectral_index=spectral_index,
                reference_freq=obs.reference_freq_hz,
                freq_min=obs.freq_min_hz,
                freq_max=obs.freq_max_hz
            )

            new_peak_flux = target_expected / correction if correction > 0 else target_expected
            expected_flux = target_expected
            peak_flux = new_peak_flux
            scaled_count += 1

        # Update row dict with scaled values and use factory method
        row_dict['peak_flux_jy'] = float(peak_flux)
        row_dict['expected_cube_flux_jy'] = float(expected_flux)
        transients.append(Transient.from_manifest_row(row_dict))

    print(f"Scaled {scaled_count}/{len(transients)} transients")

    # Update metadata
    metadata['rms_jy'] = rms
    metadata['snr_range'] = [snr_min, snr_max]
    metadata['scaled_count'] = scaled_count

    return transients, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scale transient fluxes to target SNR range')
    parser.add_argument('--manifest', required=True, help='Input manifest ECSV')
    parser.add_argument('--rms-cube', help='Baseline cube (zarr) for RMS (overrides --rms)')
    parser.add_argument('--rms', type=float, default=1.4e-04, help='RMS noise in Jy (default: 1.4e-04 from SM1R00C04_1min baseline)')
    parser.add_argument('--ms', required=True, help='Measurement Set path')
    parser.add_argument('--snr-min', type=float, default=5.0, help='Min target SNR')
    parser.add_argument('--snr-max', type=float, default=20.0, help='Max target SNR')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--output-yaml', '-o', required=True, help='Output YAML path')
    parser.add_argument('--output-manifest', '-m', help='Output manifest ECSV path')
    args = parser.parse_args()

    # --rms-cube overrides default --rms if provided

    transients, metadata = scale_fluxes(
        args.manifest, args.ms,
        snr_min=args.snr_min, snr_max=args.snr_max, seed=args.seed,
        rms=args.rms, rms_cube_path=args.rms_cube
    )

    save_transients_yaml(transients, args.output_yaml)
    print(f"Saved YAML to {args.output_yaml}")

    if args.output_manifest:
        save_manifest_ecsv(transients, args.output_manifest, metadata)
        print(f"Saved manifest to {args.output_manifest}")
