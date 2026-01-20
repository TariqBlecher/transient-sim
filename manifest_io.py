"""ECSV manifest I/O for transient simulation data."""

import pandas as pd
from pathlib import Path
from dataclasses import asdict
from astropy.table import Table


def save_manifest_ecsv(transients, output_path, metadata):
    """Save transients to ECSV with metadata in header."""
    output_path = Path(output_path).with_suffix('.ecsv')
    records = [asdict(t) for t in transients]
    table = Table.from_pandas(pd.DataFrame(records))
    table.meta = metadata
    table.write(output_path, format='ascii.ecsv', overwrite=True)
    return output_path


def load_manifest_ecsv(path):
    """Load ECSV manifest, return (DataFrame, metadata dict)."""
    table = Table.read(path, format='ascii.ecsv')
    return table.to_pandas(), dict(table.meta)
