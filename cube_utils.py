"""Zarr cube I/O and WCS utilities.

Functions for loading cube data, extracting RMS, and coordinate transforms.
"""

import numpy as np
import xarray as xr
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Tuple, Union
from pathlib import Path


def load_cube(cube_path: Union[str, Path]) -> Tuple[np.ndarray, WCS, np.ndarray]:
    """Load a zarr cube and return data, WCS, and relative times.

    Args:
        cube_path: Path to zarr cube directory

    Returns:
        Tuple of:
        - data: 3D numpy array (time, y, x), squeezed
        - wcs: Astropy WCS object for spatial coordinates
        - times_rel: 1D array of times relative to first timestep (seconds)
    """
    ds = xr.open_zarr(cube_path)
    data = ds.cube.to_numpy().squeeze()
    wcs = WCS(dict(ds.fits_header), naxis=2)

    times = ds.TIME.to_numpy()
    times_rel = times - times.min()

    return data, wcs, times_rel


def extract_rms_from_cube(cube_path: Union[str, Path]) -> float:
    """Extract mean RMS from a zarr cube.

    Args:
        cube_path: Path to zarr cube directory

    Returns:
        Mean RMS value in Jy
    """
    ds = xr.open_zarr(cube_path)
    return float(ds.rms.to_numpy().mean())


def get_pixel_coords(wcs: WCS, ra_deg: float, dec_deg: float) -> Tuple[int, int]:
    """Convert RA/Dec to pixel coordinates.

    Args:
        wcs: Astropy WCS object
        ra_deg: Right ascension in degrees
        dec_deg: Declination in degrees

    Returns:
        Tuple of (x_pixel, y_pixel) as integers
    """
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    px, py = wcs.world_to_pixel(coord)
    return int(np.round(px)), int(np.round(py))


def compute_bin_edges(times_rel: np.ndarray, integration: float = None) -> np.ndarray:
    """Compute time bin edges from relative times.

    Each bin is centered on times_rel[i] with width = integration.

    Args:
        times_rel: Array of relative times (seconds)
        integration: Integration time per bin (seconds). If None, computed
                     from median of time differences.

    Returns:
        Array of bin edges with length len(times_rel) + 1
    """
    n_times = len(times_rel)
    if integration is None:
        integration = float(np.median(np.diff(times_rel))) if n_times > 1 else float(times_rel.max())

    bin_edges = np.zeros(n_times + 1)
    for i in range(n_times):
        bin_edges[i] = times_rel[i] - integration / 2
    bin_edges[-1] = times_rel[-1] + integration / 2

    return bin_edges


def extract_lightcurve(cube: np.ndarray, px: int, py: int, aperture: int = 2) -> np.ndarray:
    """Extract lightcurve at a pixel position with aperture.

    Args:
        cube: 3D data array (time, y, x)
        px: X pixel coordinate
        py: Y pixel coordinate
        aperture: Half-width of extraction box (default 2 = 5x5 box)

    Returns:
        1D array of max flux values per time step
    """
    y_slice = slice(max(0, py - aperture), min(cube.shape[1], py + aperture + 1))
    x_slice = slice(max(0, px - aperture), min(cube.shape[2], px + aperture + 1))
    return cube[:, y_slice, x_slice].max(axis=(1, 2))
