"""Verify transient injection by comparing expected vs measured fluxes."""

import argparse
from pathlib import Path

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
        lightcurve = extract_lightcurve(cube, px, py)

        # Get measured at expected peak time bin
        measured_at_peak = float(lightcurve[time_idx])

        results.append(
            {
                "name": t["name"],
                "expected": expected,
                "measured": measured_at_peak,
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
        r["diff_sigma"] = diff / rms if rms > 0 else 0
        r["snr"] = r["measured"] / rms if rms > 0 else 0
        r["ratio"] = r["measured"] / r["expected"] if r["expected"] > 0 else 0
        r["measured_matches_expected_within_3sigma"] = r["diff_sigma"] < 3

    n_pass = sum(1 for r in valid if r["measured_matches_expected_within_3sigma"])
    print(f"\nResults: {n_pass}/{len(valid)} transients within 3 sigma of expected")

    # Also compute detection stats
    n_detected = sum(1 for r in valid if r["measured"] > 5 * rms)
    print(f"Detected (>5 sigma): {n_detected}/{len(valid)}")

    # Print results table
    print(
        "\nName         Expected   Measured   Ratio    SNR  d/sig  Shape       OK"
    )
    print("-" * 75)
    for r in valid[:15]:
        ok = "Y" if r["measured_matches_expected_within_3sigma"] else "N"
        print(
            f"{r['name']:<12} {r['expected']:>8.5f} {r['measured']:>10.5f} "
            f"{r['ratio']:>7.2f} {r['snr']:>6.1f} {r['diff_sigma']:>6.1f}  {r['shape']:<11} {ok}"
        )

    # Analyze by shape
    print("\nBy shape:")
    for shape in ["gaussian", "exponential", "step"]:
        shape_results = [r for r in valid if r["shape"] == shape]
        if shape_results:
            n_pass_shape = sum(1 for r in shape_results if r["measured_matches_expected_within_3sigma"])
            n_det_shape = sum(1 for r in shape_results if r["measured"] > 5 * rms)
            avg_ratio = np.mean([r["ratio"] for r in shape_results])
            print(
                f"  {shape}: {n_pass_shape}/{len(shape_results)} pass, "
                f"{n_det_shape} detected, avg ratio: {avg_ratio:.2f}"
            )

    # Write results to ECSV
    from astropy.table import Table

    manifest_stem = Path(manifest_path).stem
    output_path = Path(manifest_path).parent / f"{manifest_stem}_verification.ecsv"

    table = Table(rows=valid)
    table.meta = {"manifest_path": str(manifest_path), "cube_path": str(cube_path), "rms_jy": rms}
    table.write(output_path, format="ascii.ecsv", overwrite=True)
    print(f"\nResults written to {output_path}")

    # Plot lightcurves
    plot_dir = Path(manifest_path).parent / f"{manifest_stem}_lightcurves"
    plot_lightcurves(manifest_path, cube_path, plot_dir)

    return results


def flag_invalid_timeslots(manifest_path, cube_path, output_path=None):
    """Flag transients whose peak falls in invalid timeslots.

    Adds 'valid_timeslot' boolean column to manifest.

    Args:
        manifest_path: Path to input manifest ECSV
        cube_path: Path to zarr cube with nonzero array
        output_path: Output path (default: overwrites input)

    Returns:
        Tuple of (n_valid, n_invalid)
    """
    import xarray as xr
    from astropy.table import Table

    df, meta = load_manifest_ecsv(manifest_path)

    # Load cube time info
    ds = xr.open_zarr(cube_path)
    nonzero = ds["nonzero"].values.squeeze()
    times = ds.TIME.values
    times_rel = times - times.min()

    # Check each transient
    valid_flags = []
    for _, row in df.iterrows():
        time_idx = int(np.argmin(np.abs(times_rel - row["peak_time_sec"])))
        valid_flags.append(bool(nonzero[time_idx]))

    df["valid_timeslot"] = valid_flags

    # Save preserving ECSV format
    out = output_path or manifest_path
    table = Table.from_pandas(df)
    table.meta = meta
    table.write(out, format="ascii.ecsv", overwrite=True)

    n_valid = sum(valid_flags)
    n_invalid = len(valid_flags) - n_valid
    return n_valid, n_invalid


def plot_lightcurves(manifest_path, cube_path, output_dir):
    """Plot lightcurves for all transients and save to folder.

    Args:
        manifest_path: Path to manifest ECSV file
        cube_path: Path to zarr cube
        output_dir: Directory to save individual plots
    """
    import matplotlib.pyplot as plt

    df, _ = load_manifest_ecsv(manifest_path)
    transients = df.to_dict("records")

    cube, cube_wcs, times_rel = load_cube(cube_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for t in transients:
        px, py = get_pixel_coords(cube_wcs, t["ra_deg"], t["dec_deg"])
        lightcurve = extract_lightcurve(cube, px, py)

        name = t["name"]
        shape = t["shape"]
        peak_time = t["peak_time_sec"]
        expected = t["expected_cube_flux_jy"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(times_rel, lightcurve, "o-", markersize=3, label="Measured")
        ax.axvline(peak_time, color="red", linestyle="--", alpha=0.7, label="Peak time")
        ax.axhline(expected, color="green", linestyle=":", alpha=0.7, label="Expected")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Flux (Jy)")
        ax.set_title(f"{name} ({shape})")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / f"{name}.png", dpi=150)
        plt.close()

    print(f"Lightcurves saved to {output_dir}/ ({len(transients)} plots)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify transient injection")
    parser.add_argument("--manifest", required=True, help="Manifest ECSV path")
    parser.add_argument("--cube", required=True, help="Zarr cube path")
    parser.add_argument(
        "--flag-timeslots",
        action="store_true",
        help="Add valid_timeslot column to manifest based on cube nonzero array",
    )
    args = parser.parse_args()

    if args.flag_timeslots:
        n_valid, n_invalid = flag_invalid_timeslots(args.manifest, args.cube)
        print(f"Flagged {n_invalid} transients in invalid timeslots ({n_valid} valid)")
    else:
        verify_transients(args.manifest, args.cube)
