#!/usr/bin/env python3
"""Simple zarr cube to FITS converter."""

import sys
import os
import numpy as np
import xarray
from astropy.io import fits


from typing import Optional

def zarr_to_fits(zarr_path: str, out_fits: Optional[str] = None, time_chunk: int = 16):
    """Convert a zarr cube to FITS format using streaming.

    Squeezes degenerate FREQ dimension (size=1) to produce 4D CARTA-compatible output.
    """

    ds = xarray.open_zarr(zarr_path, chunks={"TIME": time_chunk})

    if "cube" not in ds.data_vars:
        raise RuntimeError(f"{zarr_path} does not contain a 'cube' variable")

    data = ds.data_vars["cube"]
    print(f"Cube shape: {data.shape}, dims: {data.dims}")

    # Output filename
    if out_fits is None:
        out_fits = zarr_path.rstrip("/").replace(".zarr", ".fits")

    # FITS header from zarr attrs
    hdr = dict(ds.attrs.get("fits_header", {}))
    hdr.pop("WCSAXES", None)

    # Detect if we need to squeeze FREQ dimension
    # 5D: (STOKES, FREQ, TIME, Y, X) -> squeeze FREQ if size=1
    squeeze_freq = len(data.shape) == 5 and data.shape[1] == 1

    if squeeze_freq:
        out_shape = (data.shape[0], data.shape[2], data.shape[3], data.shape[4])
        print(f"Squeezing FREQ dimension: {data.shape} -> {out_shape}")
    else:
        out_shape = data.shape

    hdr["NAXIS"] = len(out_shape)
    hdr["BITPIX"] = -32  # float32

    # Build ordered header with standard keywords first
    kws = ["SIMPLE", "BITPIX", "NAXIS"] + [
        f"NAXIS{i}" for i in range(1, len(out_shape) + 1)
    ]
    ordered_header = {}
    for kw in kws:
        if kw in hdr:
            ordered_header[kw] = hdr.pop(kw)
    ordered_header.update(hdr)

    # Set NAXIS values from output shape (reversed for FITS)
    for i, sz in enumerate(reversed(out_shape)):
        ordered_header[f"NAXIS{i + 1}"] = sz

    ordered_header["WCSAXES"] = len(out_shape)

    # Remap WCS keywords if squeezing FREQ
    # Original 5D FITS axes: 1=X, 2=Y, 3=TIME, 4=FREQ, 5=STOKES
    # Output 4D FITS axes:   1=X, 2=Y, 3=TIME, 4=STOKES
    if squeeze_freq:
        wcs_kws = ["CTYPE", "CRPIX", "CRVAL", "CDELT", "CUNIT"]
        # Move axis 5 (STOKES) to axis 4
        for kw in wcs_kws:
            if f"{kw}5" in ordered_header:
                ordered_header[f"{kw}4"] = ordered_header.pop(f"{kw}5")
            elif f"{kw}4" in ordered_header:
                # Remove old FREQ axis keywords
                ordered_header.pop(f"{kw}4")
        # Remove any leftover axis 5 keywords
        for kw in wcs_kws + ["NAXIS"]:
            ordered_header.pop(f"{kw}5", None)

    print(f"Writing to {out_fits}")

    if os.path.exists(out_fits):
        os.unlink(out_fits)

    shdu = fits.StreamingHDU(out_fits, fits.Header(ordered_header))

    # Stream write in chunks
    if len(data.shape) == 5:  # (stokes, freq, time, y, x)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for t in range(0, data.shape[2], time_chunk):
                    chunk = (
                        data[i, j, t : t + time_chunk].data.compute().astype(np.float32)
                    )
                    shdu.write(chunk)
                    print(f"  Written stokes={i}, freq={j}, time={t}-{t + len(chunk)}")
    elif len(data.shape) == 4:  # (stokes, time, y, x)
        for i in range(data.shape[0]):
            for t in range(0, data.shape[1], time_chunk):
                chunk = data[i, t : t + time_chunk].data.compute().astype(np.float32)
                shdu.write(chunk)
                print(f"  Written stokes={i}, time={t}-{t + len(chunk)}")
    else:
        raise ValueError(f"Unexpected shape: {data.shape}")

    shdu.close()
    print(f"Done: {out_fits}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: zarr_to_fits_simple.py <zarr_path> [output.fits]")
        sys.exit(1)

    zarr_path = sys.argv[1]
    out_fits = sys.argv[2] if len(sys.argv) > 2 else None
    zarr_to_fits(zarr_path, out_fits)
