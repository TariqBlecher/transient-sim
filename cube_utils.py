"""Zarr cube I/O utilities.

Functions for extracting data from zarr cubes.
"""

import xarray as xr
from typing import Union
from pathlib import Path


def extract_rms_from_cube(cube_path: Union[str, Path]) -> float:
    """Extract mean RMS from a zarr cube.

    Args:
        cube_path: Path to zarr cube directory

    Returns:
        Mean RMS value in Jy
    """
    ds = xr.open_zarr(cube_path)
    return float(ds.rms.to_numpy().mean())
