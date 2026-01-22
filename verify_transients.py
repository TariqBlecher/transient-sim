"""Verify transient injection by comparing expected vs measured fluxes."""

import argparse
import numpy as np

from manifest_io import load_manifest_ecsv
from verify_utils import load_cube, get_pixel_coords, extract_lightcurve


def verify_transients(manifest_path, cube_path):
    """Verify transient injection against expected fluxes.

    Args:
        manifest_path: Path to manifest ECSV file
        cube_path: Path to zarr cube

    Returns:
        List of result dicts per transient
    """
    df, _ = load_manifest_ecsv(manifest_path)
    transients = df.to_dict("records")
    print(f"Loaded {len(transients)} transients from manifest")

    # Load cube
    cube, cube_wcs, times_rel = load_cube(cube_path)
    print(f"Cube shape: {cube.shape}, {len(times_rel)} time steps")

    # Check each transient
    results = []
    for t in transients:
        px, py = get_pixel_coords(cube_wcs, t["ra_deg"], t["dec_deg"])

        if not (0 <= px < cube.shape[2] and 0 <= py < cube.shape[1]):
            results.append({"name": t["name"], "status": "out_of_bounds"})
            continue

        peak_time = t["peak_time_sec"]
        shape = t["shape"]
        expected = t["expected_cube_flux_jy"]

        # Find time bin containing peak
        time_idx = int(np.argmin(np.abs(times_rel - peak_time)))

        # Extract lightcurve
        lightcurve = extract_lightcurve(cube, px, py, aperture=2)

        # Get measured at expected peak time bin
        measured_at_peak = float(lightcurve[time_idx])

        # Also get max measured across all bins (ignore NaNs)
        max_measured = float(np.nanmax(lightcurve))
        max_time_idx = int(np.nanargmax(lightcurve))

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
            }
        )

    # Get noise from rms array - need to reopen for rms
    import xarray as xr

    ds = xr.open_zarr(cube_path)
    rms = float(np.nanmean(ds.rms.to_numpy()))
    print(f"RMS noise: {rms:.6f} Jy")

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
        "\nName         Expected   Measured   MaxMeas    Ratio   n_sig  Shape       OK"
    )
    print("-" * 82)
    for r in valid[:15]:
        ok = "Y" if r["pass"] else "N"
        print(
            f"{r['name']:<12} {r['expected']:>8.5f} {r['measured']:>10.5f} {r['max_measured']:>9.5f} "
            f"{r['ratio']:>7.2f} {r['n_sigma']:>6.1f}  {r['shape']:<11} {ok}"
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
        expected = t["expected_cube_flux_jy"]
        max_meas = float(np.nanmax(lightcurve))
        max_idx = int(np.nanargmax(lightcurve))

        print(f"{name} ({shape})")
        print(f"  peak_time: {peak_time:.2f}s, duration: {duration:.2f}s")
        print(f"  expected cube flux: {expected:.5f} Jy")
        print(f"  lightcurve (non-NaN count: {np.sum(~np.isnan(lightcurve))}): min={np.nanmin(lightcurve):.5f}, max={max_meas:.5f}")
        print(f"  max measured: {max_meas:.5f} Jy at t_idx={max_idx}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify transient injection")
    parser.add_argument("--manifest", required=True, help="Manifest ECSV path")
    parser.add_argument("--cube", required=True, help="Zarr cube path")
    parser.add_argument(
        "--lightcurves", action="store_true", help="Show detailed lightcurves"
    )
    args = parser.parse_args()

    if args.lightcurves:
        show_lightcurves(args.manifest, args.cube)
    else:
        verify_transients(args.manifest, args.cube)
