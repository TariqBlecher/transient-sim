"""Scale transient fluxes to achieve target SNR range in output cubes."""

import argparse
import numpy as np
import yaml
from pathlib import Path
from manifest_io import load_manifest_ecsv, save_manifest_ecsv
from transient_sim import (
    extract_rms_from_cube,
    compute_expected_flux,
    Transient,
    ObservationParams,
)


def scale_fluxes(manifest_path, rms_cube_path, ms_path, snr_min=5.0, snr_max=20.0, seed=None):
    """
    Scale transient fluxes so expected cube flux falls within SNR range.

    For each transient:
    - Compute expected_cube_flux from current peak_flux
    - Calculate SNR = expected_cube_flux / rms
    - If SNR < snr_min or SNR > snr_max, scale peak_flux to hit random target in range
    """
    rng = np.random.default_rng(seed)

    # Load inputs
    df, metadata = load_manifest_ecsv(manifest_path)
    rms = extract_rms_from_cube(rms_cube_path)
    obs = ObservationParams.from_ms(ms_path)

    print(f"Loaded {len(df)} transients")
    print(f"RMS from cube: {rms:.4f} Jy")
    print(f"Target SNR range: {snr_min}-{snr_max}")

    scaled_count = 0
    transients = []

    for _, row in df.iterrows():
        peak_flux = row['peak_flux_jy']
        peak_time = row['peak_time_sec']
        duration = row['duration_sec']
        shape = row['shape']
        spectral_index = row['spectral_index']

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

        transients.append(Transient(
            name=row['name'],
            ra_deg=row['ra_deg'],
            dec_deg=row['dec_deg'],
            peak_time_sec=peak_time,
            duration_sec=duration,
            shape=shape,
            peak_flux_jy=float(peak_flux),
            spectral_index=spectral_index,
            reference_freq_hz=row['reference_freq_hz'],
            periodic=row.get('periodic', False),
            period_sec=row.get('period_sec'),
            expected_cube_flux_jy=float(expected_flux)
        ))

    print(f"Scaled {scaled_count}/{len(transients)} transients")

    # Update metadata
    metadata['rms_jy'] = rms
    metadata['snr_range'] = [snr_min, snr_max]
    metadata['scaled_count'] = scaled_count

    return transients, metadata


def save_yaml(transients, output_path):
    """Save transients to YAML format for pfb."""
    output_path = Path(output_path)
    data = {'transients': [t.to_yaml_dict() for t in transients]}
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scale transient fluxes to target SNR range')
    parser.add_argument('--manifest', required=True, help='Input manifest ECSV')
    parser.add_argument('--rms-cube', required=True, help='Baseline cube (zarr) for RMS')
    parser.add_argument('--ms', required=True, help='Measurement Set path')
    parser.add_argument('--snr-min', type=float, default=5.0, help='Min target SNR')
    parser.add_argument('--snr-max', type=float, default=20.0, help='Max target SNR')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--output-yaml', '-o', required=True, help='Output YAML path')
    parser.add_argument('--output-manifest', '-m', help='Output manifest ECSV path')
    args = parser.parse_args()

    transients, metadata = scale_fluxes(
        args.manifest, args.rms_cube, args.ms,
        snr_min=args.snr_min, snr_max=args.snr_max, seed=args.seed
    )

    save_yaml(transients, args.output_yaml)
    print(f"Saved YAML to {args.output_yaml}")

    if args.output_manifest:
        save_manifest_ecsv(transients, args.output_manifest, metadata)
        print(f"Saved manifest to {args.output_manifest}")
