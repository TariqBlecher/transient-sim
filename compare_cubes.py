"""Compare cubes with and without transient injection."""

import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from manifest_io import load_manifest_ecsv

manifest_path = "/vault2-tina/blecher/smc_notebooks/transient_sims/test_pipeline_manifest.ecsv"
cube_with = "/vault2-tina/blecher/SMC/transient_sim/pfb_approach/hci_pipeline_test"
cube_without = "/vault2-tina/blecher/SMC/transient_sim/pfb_approach/hci_no_transients"

df, metadata = load_manifest_ecsv(manifest_path)
transients = df.to_dict('records')

ds_with = xr.open_zarr(cube_with)
ds_without = xr.open_zarr(cube_without)

cube_w = ds_with.cube.to_numpy().squeeze()
cube_wo = ds_without.cube.to_numpy().squeeze()
cube_wcs = WCS(dict(ds_with.fits_header), naxis=2)

# Difference cube = injected transients only
diff = cube_w - cube_wo

times = ds_with.TIME.to_numpy()
times_rel = times - times.min()

print(f"Cube shape: {cube_w.shape}")
print(f"Time bins: {np.round(times_rel, 1)}")
print(f"Diff cube stats: min={diff.min():.4f}, max={diff.max():.4f}, std={diff.std():.4f}")
print()

# Check each transient in the difference cube
print("Transient verification from difference cube:")
print("Name         Shape        PeakFlux  MaxDiff   Ratio   t_idx  Status")
print("-" * 75)

results = []
for t in transients:
    coord = SkyCoord(ra=t["ra_deg"]*u.deg, dec=t["dec_deg"]*u.deg)
    px, py = cube_wcs.world_to_pixel(coord)
    px, py = int(np.round(px)), int(np.round(py))

    if not (0 <= px < diff.shape[2] and 0 <= py < diff.shape[1]):
        continue

    aperture = 3
    y_slice = slice(max(0, py-aperture), min(diff.shape[1], py+aperture+1))
    x_slice = slice(max(0, px-aperture), min(diff.shape[2], px+aperture+1))

    # Get max in the difference cube at this position
    diff_patch = diff[:, y_slice, x_slice]
    max_diff = float(diff_patch.max())
    max_idx = np.unravel_index(diff_patch.argmax(), diff_patch.shape)
    time_idx = max_idx[0]

    peak_flux = t["peak_flux_jy"]
    ratio = max_diff / peak_flux if peak_flux > 0 else 0

    # Check if within 20%
    status = "OK" if 0.8 <= ratio <= 1.2 else ("LOW" if ratio < 0.8 else "HIGH")

    name = t["name"]
    shape = t["shape"]
    print(f"{name:<12} {shape:<12} {peak_flux:>8.2f} {max_diff:>8.3f} {ratio:>7.2f} {time_idx:>6}  {status}")

    results.append({"name": name, "ratio": ratio, "status": status, "shape": shape})

# Summary
print()
n_ok = sum(1 for r in results if r["status"] == "OK")
print(f"Within 20%: {n_ok}/{len(results)}")

for shape in ["gaussian", "exponential", "step"]:
    shape_res = [r for r in results if r["shape"] == shape]
    n_ok_shape = sum(1 for r in shape_res if r["status"] == "OK")
    avg_ratio = np.mean([r["ratio"] for r in shape_res])
    print(f"  {shape}: {n_ok_shape}/{len(shape_res)} OK, avg ratio: {avg_ratio:.2f}")
