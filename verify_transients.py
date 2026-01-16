"""Verify transient injection by comparing expected vs measured fluxes."""

import numpy as np
from manifest_io import load_manifest_ecsv
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
import argparse
from scipy import integrate


def compute_spectral_correction(spectral_index, reference_freq, freq_min, freq_max):
    """
    Compute the flux correction factor for spectral index.

    pfb applies: fprofile(ν) = peak_flux * (ν/ν_ref)^α
    The cube averages over frequency, so effective flux = mean(fprofile) over band.

    Returns the ratio of band-averaged flux to peak_flux.
    """
    alpha = spectral_index
    nu_ref = reference_freq
    nu_min = freq_min
    nu_max = freq_max

    if abs(alpha) < 1e-10:
        # Flat spectrum
        return 1.0

    if abs(alpha + 1) < 1e-10:
        # Special case: α = -1 (logarithmic integral)
        return nu_ref * np.log(nu_max / nu_min) / (nu_max - nu_min)

    # General case: integrate power law
    # ∫ (ν/ν_ref)^α dν = ν_ref^(-α) * ν^(α+1) / (α+1)
    integral = (nu_max**(alpha + 1) - nu_min**(alpha + 1)) / (alpha + 1)
    average = integral / (nu_max - nu_min)
    # Divide by ν_ref^α to get the correction factor
    correction = average / (nu_ref ** alpha)

    return correction


def compute_expected_flux(peak_flux, peak_time, duration, shape, bin_start, bin_end,
                          spectral_index=0.0, reference_freq=None, freq_min=None, freq_max=None):
    """
    Compute expected measured flux in a time bin, accounting for shape and spectral index.

    Matches pfb-imaging implementation in pfb/utils/transients.py:
    - gaussian: sigma = duration (NOT FWHM-based)
    - exponential: tau = duration
    - step: active for [peak_time, peak_time + duration]
    - spectral index: flux(ν) = peak_flux * (ν/ν_ref)^α, averaged over band
    """
    bin_width = bin_end - bin_start

    # Apply spectral correction if frequency info provided
    if reference_freq is not None and freq_min is not None and freq_max is not None:
        spec_correction = compute_spectral_correction(spectral_index, reference_freq, freq_min, freq_max)
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
            return effective_peak_flux * np.exp(-((t - peak_time) ** 2) / (2 * sigma ** 2))

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


def verify_transients(manifest_path, cube_path, ms_path=None):
    df, metadata = load_manifest_ecsv(manifest_path)
    transients = df.to_dict('records')
    print(f"Loaded {len(transients)} transients from manifest")

    # Load cube
    ds = xr.open_zarr(cube_path)
    cube = ds.cube.to_numpy().squeeze()
    cube_wcs = WCS(dict(ds.fits_header), naxis=2)

    print(f"Cube shape: {cube.shape}")

    # Get frequency range from MS or manifest
    freq_min, freq_max = None, None
    if ms_path:
        from casacore.tables import table
        with table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True, ack=False) as spw:
            freqs = spw.getcol("CHAN_FREQ")[0]
            freq_min = float(freqs.min())
            freq_max = float(freqs.max())
        print(f"Frequency range: {freq_min/1e9:.3f} - {freq_max/1e9:.3f} GHz")
    elif metadata.get("ms_path"):
        ms = metadata["ms_path"]
        try:
            from casacore.tables import table
            with table(f"{ms}/SPECTRAL_WINDOW", readonly=True, ack=False) as spw:
                freqs = spw.getcol("CHAN_FREQ")[0]
                freq_min = float(freqs.min())
                freq_max = float(freqs.max())
            print(f"Frequency range: {freq_min/1e9:.3f} - {freq_max/1e9:.3f} GHz")
        except Exception:
            print("Warning: Could not get frequency range from MS")

    # Get time array and compute bin edges (bins are CENTERED on times_rel)
    times = ds.TIME.to_numpy()
    times_rel = times - times.min()
    n_times = len(times_rel)
    integration = float(np.median(np.diff(times_rel))) if n_times > 1 else float(times_rel.max())
    print(f"Time steps: {n_times}, integration: {integration:.2f}s")

    # Bin edges: each bin is centered on times_rel[i], width = integration
    bin_edges = np.zeros(n_times + 1)
    for i in range(n_times):
        bin_edges[i] = times_rel[i] - integration / 2
    bin_edges[-1] = times_rel[-1] + integration / 2

    # Check each transient
    results = []
    for t in transients:
        coord = SkyCoord(ra=t["ra_deg"]*u.deg, dec=t["dec_deg"]*u.deg)
        px, py = cube_wcs.world_to_pixel(coord)
        px, py = int(np.round(px)), int(np.round(py))

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

        aperture = 2
        y_slice = slice(max(0, py-aperture), min(cube.shape[1], py+aperture+1))
        x_slice = slice(max(0, px-aperture), min(cube.shape[2], px+aperture+1))
        lightcurve = cube[:, y_slice, x_slice].max(axis=(1, 2))

        # Get measured at expected peak time bin
        measured_at_peak = float(lightcurve[time_idx])

        # Also get max measured across all bins
        max_measured = float(lightcurve.max())
        max_time_idx = int(np.argmax(lightcurve))

        # Compute expected flux accounting for shape, integration, and spectral index
        bin_start = bin_edges[time_idx]
        bin_end = bin_edges[time_idx + 1]
        expected = compute_expected_flux(
            peak_flux, peak_time, duration, shape, bin_start, bin_end,
            spectral_index=spectral_index,
            reference_freq=reference_freq,
            freq_min=freq_min,
            freq_max=freq_max
        )

        results.append({
            "name": t["name"],
            "expected": expected,
            "measured": measured_at_peak,
            "max_measured": max_measured,
            "max_time_idx": max_time_idx,
            "peak_time": peak_time,
            "time_idx": time_idx,
            "shape": shape,
            "peak_flux": peak_flux,
            "spectral_index": spectral_index
        })

    # Get noise from rms array
    rms = float(ds.rms.to_numpy().mean())
    print(f"RMS noise: {rms:.4f} Jy")

    # Evaluate results
    valid = [r for r in results if "expected" in r]
    for r in valid:
        diff = abs(r["measured"] - r["expected"])
        r["diff"] = diff
        r["n_sigma"] = diff / rms if rms > 0 else 0
        r["ratio"] = r["measured"] / r["expected"] if r["expected"] > 0 else 0
        # Pass if within 3σ of expected
        r["pass"] = r["n_sigma"] < 3

    n_pass = sum(1 for r in valid if r["pass"])
    print(f"\nResults: {n_pass}/{len(valid)} transients within 3σ of expected")

    # Also compute detection stats (is there significant flux at the position?)
    n_detected = sum(1 for r in valid if r["measured"] > 5 * rms)
    print(f"Detected (>5σ): {n_detected}/{len(valid)}")

    # Print results table
    print("\nName         Expected   Measured  MaxMeas   Ratio    nσ   Shape       OK")
    print("-" * 80)
    for r in valid[:15]:
        ok = "Y" if r["pass"] else "N"
        print(f"{r['name']:<12} {r['expected']:>8.3f} {r['measured']:>10.3f} {r['max_measured']:>8.3f} "
              f"{r['ratio']:>7.2f} {r['n_sigma']:>6.1f} {r['shape']:<12} {ok}")

    # Analyze by shape
    print("\nBy shape:")
    for shape in ["gaussian", "exponential", "step"]:
        shape_results = [r for r in valid if r["shape"] == shape]
        if shape_results:
            n_pass_shape = sum(1 for r in shape_results if r["pass"])
            n_det_shape = sum(1 for r in shape_results if r["measured"] > 5 * rms)
            avg_ratio = np.mean([r["ratio"] for r in shape_results])
            avg_sigma = np.mean([r["n_sigma"] for r in shape_results])
            print(f"  {shape}: {n_pass_shape}/{len(shape_results)} pass, "
                  f"{n_det_shape} detected, avg ratio: {avg_ratio:.2f}")

    return results


def show_lightcurves(manifest_path, cube_path, n=5):
    """Show detailed lightcurves for first n transients."""
    df, _ = load_manifest_ecsv(manifest_path)
    transients = df.to_dict('records')
    ds = xr.open_zarr(cube_path)
    cube = ds.cube.to_numpy().squeeze()
    cube_wcs = WCS(dict(ds.fits_header), naxis=2)
    times = ds.TIME.to_numpy()
    times_rel = times - times.min()

    print("Time bins:", times_rel)
    print()

    for t in transients[:n]:
        coord = SkyCoord(ra=t["ra_deg"]*u.deg, dec=t["dec_deg"]*u.deg)
        px, py = cube_wcs.world_to_pixel(coord)
        px, py = int(np.round(px)), int(np.round(py))

        aperture = 2
        y_slice = slice(max(0, py-aperture), min(cube.shape[1], py+aperture+1))
        x_slice = slice(max(0, px-aperture), min(cube.shape[2], px+aperture+1))
        lightcurve = cube[:, y_slice, x_slice].max(axis=(1, 2))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cube", required=True)
    parser.add_argument("--ms", help="Path to Measurement Set (for frequency info)")
    parser.add_argument("--lightcurves", action="store_true", help="Show detailed lightcurves")
    args = parser.parse_args()

    if args.lightcurves:
        show_lightcurves(args.manifest, args.cube)
    else:
        verify_transients(args.manifest, args.cube, ms_path=args.ms)
