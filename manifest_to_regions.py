#!/usr/bin/env python3
"""Convert transient manifest ECSV to DS9 region file."""

import argparse
from pathlib import Path
import pandas as pd
from manifest_io import load_manifest_ecsv


def format_label(row):
    """Format all transient properties into a compact label string."""
    parts = [
        row["name"],
        f"shape={row['shape']}",
        f"flux={row['peak_flux_jy']:.3f}Jy",
        f"t={row['peak_time_sec']:.1f}s",
        f"dur={row['duration_sec']:.1f}s",
        f"alpha={row['spectral_index']:.2f}",
    ]
    if row.get("periodic"):
        parts.append(f"P={row['period_sec']:.1f}s")
    if pd.notna(row.get("expected_cube_flux_jy")):
        parts.append(f"exp={row['expected_cube_flux_jy']:.4f}Jy")
    return " | ".join(parts)


def manifest_to_regions(manifest_path, output_path, radius=10.0):
    """Convert manifest ECSV to DS9 region file."""
    df, metadata = load_manifest_ecsv(manifest_path)

    lines = [
        "# Region file format: DS9 version 4.0",
        'global color=green dashlist=8 3 width=1 font="helvetica 14 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=0 move=0 delete=1 include=1 source=1',
        "fk5",
    ]

    for _, row in df.iterrows():
        label = format_label(row)
        lines.append(
            f'circle({row["ra_deg"]:.6f}, {row["dec_deg"]:.6f}, {radius}") '
            f"# text={{{label}}}"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Convert transient manifest ECSV to DS9 region file"
    )
    parser.add_argument("--manifest", required=True, help="Input manifest ECSV file")
    parser.add_argument(
        "--output", "-o", help="Output .reg file (default: input with .reg extension)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=10.0,
        help="Circle radius in arcsec (default: 10)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(args.manifest).with_suffix(".reg"))

    count = manifest_to_regions(args.manifest, args.output, args.radius)
    print(f"Wrote {count} regions to {args.output}")


if __name__ == "__main__":
    main()
