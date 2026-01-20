"""Verify transient injection by comparing expected vs measured fluxes."""

import argparse
import numpy as np

from manifest_io import load_manifest_ecsv
from flux_utils import compute_expected_flux
from verify_utils import (
    load_cube,
    get_pixel_coords,
    extract_lightcurve,
    compute_bin_edges,
    get_frequency_range_from_ms,
)


def verify_transients(manifest_path, cube_path, ms_path=None):
    """Verify transient injection against expected fluxes.

    Args:
        manifest_path: Path to manifest ECSV file
        cube_path: Path to zarr cube
        ms_path: Optional path to MS for frequency info

    Returns:
        List of result dicts per transient
    """
    df, metadata = load_manifest_ecsv(manifest_path)
    transients = df.to_dict("records")
    print(f"Loaded {len(transients)} transients from manifest")

    # Load cube
    cube, cube_wcs, times_rel = load_cube(cube_path)
    print(f"Cube shape: {cube.shape}")

    # Get frequency range from MS or manifest
    freq_min, freq_max = None, None
    if ms_path:
        try:
            freq_min, freq_max, _ = get_frequency_range_from_ms(ms_path)
            print(f"Frequency range: {freq_min / 1e9:.3f} - {freq_max / 1e9:.3f} GHz")
        except Exception as e:
            print(f"Warning: Could not get frequency range from MS '{ms_path}': {e}")
            print(
                "         Spectral correction will be SKIPPED - results may be inaccurate!"
            )
    elif metadata.get("ms_path"):
        try:
            freq_min, freq_max, _ = get_frequency_range_from_ms(metadata["ms_path"])
            print(f"Frequency range: {freq_min / 1e9:.3f} - {freq_max / 1e9:.3f} GHz")
        except Exception as e:
            print(
                f"Warning: Could not get frequency range from MS '{metadata['ms_path']}': {e}"
            )
            print(
                "         Spectral correction will be SKIPPED - results may be inaccurate!"
            )
    else:
        print("Warning: No MS path provided - spectral correction will be SKIPPED")

    # Compute time bins
    n_times = len(times_rel)
    integration = (
        float(np.median(np.diff(times_rel))) if n_times > 1 else float(times_rel.max())
    )
    bin_edges = compute_bin_edges(times_rel, integration)
    print(f"Time steps: {n_times}, integration: {integration:.2f}s")

    # Check each transient
    results = []
    for t in transients:
        px, py = get_pixel_coords(cube_wcs, t["ra_deg"], t["dec_deg"])

        if not (0 <= px < cube.shape[2] and 0 <= py < cube.shape[1]):
            results.append({"name": t["name"], "status": "out_of_bounds"})
            continue

        peak_time = t["peak_time_sec"]
        duration = t["duration_sec"]
        shape = t["shape"]
        peak_flux = t["peak_flux_jy"]
        spectral_index = t.get("spectral_index", 0.0)
        reference_freq = t.get("reference_freq_hz", None)

        # Find time bin containing peak
        time_idx = int(np.argmin(np.abs(times_rel - peak_time)))

        # Extract lightcurve
        lightcurve = extract_lightcurve(cube, px, py, aperture=2)

        # Get measured at expected peak time bin
        measured_at_peak = float(lightcurve[time_idx])

        # Also get max measured across all bins
        max_measured = float(lightcurve.max())
        max_time_idx = int(np.argmax(lightcurve))

        # Compute expected flux accounting for shape, integration, and spectral index
        bin_start = bin_edges[time_idx]
        bin_end = bin_edges[time_idx + 1]
        expected = compute_expected_flux(
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

        results.append(
            {
                "name": t["name"],
                "expected": expected,
                "measured": measured_at_peak,
                "max_measured": max_measured,
                "max_time_idx": max_time_idx,
                "peak_time": peak_time,
                "time_idx": time_idx,
                "shape": shape,
                "peak_flux": peak_flux,
                "spectral_index": spectral_index,
            }
        )

    # Get noise from rms array - need to reopen for rms
    import xarray as xr

    ds = xr.open_zarr(cube_path)
    rms = float(ds.rms.to_numpy().mean())
    print(f"RMS noise: {rms:.4f} Jy")

    # Evaluate results
    valid = [r for r in results if "expected" in r]
    for r in valid:
        diff = abs(r["measured"] - r["expected"])
        r["diff"] = diff
        r["n_sigma"] = diff / rms if rms > 0 else 0
        r["ratio"] = r["measured"] / r["expected"] if r["expected"] > 0 else 0
        # Pass if within 3 sigma of expected
        r["pass"] = r["n_sigma"] < 3

    n_pass = sum(1 for r in valid if r["pass"])
    print(f"\nResults: {n_pass}/{len(valid)} transients within 3 sigma of expected")

    # Also compute detection stats
    n_detected = sum(1 for r in valid if r["measured"] > 5 * rms)
    print(f"Detected (>5 sigma): {n_detected}/{len(valid)}")

    # Print results table
    print(
        "\nName         Expected   Measured  MaxMeas   Ratio    n_sig  Shape       OK"
    )
    print("-" * 80)
    for r in valid[:15]:
        ok = "Y" if r["pass"] else "N"
        print(
            f"{r['name']:<12} {r['expected']:>8.3f} {r['measured']:>10.3f} {r['max_measured']:>8.3f} "
            f"{r['ratio']:>7.2f} {r['n_sigma']:>6.1f} {r['shape']:<12} {ok}"
        )

    # Analyze by shape
    print("\nBy shape:")
    for shape in ["gaussian", "exponential", "step"]:
        shape_results = [r for r in valid if r["shape"] == shape]
        if shape_results:
            n_pass_shape = sum(1 for r in shape_results if r["pass"])
            n_det_shape = sum(1 for r in shape_results if r["measured"] > 5 * rms)
            avg_ratio = np.mean([r["ratio"] for r in shape_results])
            print(
                f"  {shape}: {n_pass_shape}/{len(shape_results)} pass, "
                f"{n_det_shape} detected, avg ratio: {avg_ratio:.2f}"
            )

    return results


def show_lightcurves(manifest_path, cube_path, n=5):
    """Show detailed lightcurves for first n transients."""
    df, _ = load_manifest_ecsv(manifest_path)
    transients = df.to_dict("records")

    cube, cube_wcs, times_rel = load_cube(cube_path)

    print("Time bins:", times_rel)
    print()

    for t in transients[:n]:
        px, py = get_pixel_coords(cube_wcs, t["ra_deg"], t["dec_deg"])
        lightcurve = extract_lightcurve(cube, px, py, aperture=2)

        name = t["name"]
        shape = t["shape"]
        peak_time = t["peak_time_sec"]
        duration = t["duration_sec"]
        expected = t["peak_flux_jy"]
        max_meas = float(lightcurve.max())
        max_idx = int(np.argmax(lightcurve))

        print(f"{name} ({shape})")
        print(f"  peak_time: {peak_time:.2f}s, duration: {duration:.2f}s")
        print(f"  expected flux: {expected:.3f} Jy")
        print(f"  lightcurve: {np.round(lightcurve, 3)}")
        print(f"  max measured: {max_meas:.3f} Jy at t_idx={max_idx}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify transient injection")
    parser.add_argument("--manifest", required=True, help="Manifest ECSV path")
    parser.add_argument("--cube", required=True, help="Zarr cube path")
    parser.add_argument("--ms", help="Path to Measurement Set (for frequency info)")
    parser.add_argument(
        "--lightcurves", action="store_true", help="Show detailed lightcurves"
    )
    args = parser.parse_args()

    if args.lightcurves:
        show_lightcurves(args.manifest, args.cube)
    else:
        verify_transients(args.manifest, args.cube, ms_path=args.ms)
