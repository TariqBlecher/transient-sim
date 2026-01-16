"""Verify simple transient injection."""

import yaml
import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u

yaml_path = "/vault2-tina/blecher/smc_notebooks/transient_sims/test_simple.yaml"
cube_with = "/vault2-tina/blecher/SMC/transient_sim/pfb_approach/hci_simple_test"
cube_without = "/vault2-tina/blecher/SMC/transient_sim/pfb_approach/hci_no_transients"

with open(yaml_path) as f:
    transients = yaml.safe_load(f)["transients"]

ds_with = xr.open_zarr(cube_with)
ds_without = xr.open_zarr(cube_without)
diff = ds_with.cube.to_numpy().squeeze() - ds_without.cube.to_numpy().squeeze()
cube_wcs = WCS(dict(ds_with.fits_header), naxis=2)

times = ds_with.TIME.to_numpy()
times_rel = times - times.min()

print(f"Transients: {len(transients)}")
print(f"Time bins: {np.round(times_rel, 1)}")
print()

print("Name       PeakFlux  MaxDiff   Ratio   TimeBin")
print("-" * 55)

ratios = []
for t in transients:
    coord = SkyCoord(ra=t["position"]["ra"]*u.deg, dec=t["position"]["dec"]*u.deg)
    px, py = cube_wcs.world_to_pixel(coord)
    px, py = int(np.round(px)), int(np.round(py))

    aperture = 3
    y_slice = slice(max(0, py-aperture), min(diff.shape[1], py+aperture+1))
    x_slice = slice(max(0, px-aperture), min(diff.shape[2], px+aperture+1))

    patch = diff[:, y_slice, x_slice]
    max_diff = float(patch.max())
    max_idx = np.unravel_index(patch.argmax(), patch.shape)[0]

    peak_flux = t["frequency"]["peak_flux"]
    ratio = max_diff / peak_flux

    name = t["name"]
    print(f"{name:<10} {peak_flux:>8.2f} {max_diff:>8.3f} {ratio:>7.2f} {max_idx:>7}")
    ratios.append(ratio)

print()
print(f"Mean ratio: {np.mean(ratios):.2f}")
print(f"Std ratio:  {np.std(ratios):.2f}")
print(f"Within 20%: {sum(0.8 <= r <= 1.2 for r in ratios)}/{len(ratios)}")
