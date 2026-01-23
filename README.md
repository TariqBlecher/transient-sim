# Radio Transient Simulation Framework

A modular framework for generating synthetic radio transients and injecting them into visibility data.

## Overview

This framework generates realistic radio transients with configurable temporal profiles (gaussian, exponential, step) and spectral indices, then injects them into measurement sets using the `pfb hci` imaging tool. SNR-based flux scaling ensures transients fall within detectable ranges, making the simulations meaningful for detection algorithm validation.

## Installation

**Dependencies:**
- numpy, scipy, xarray (zarr), astropy, pyyaml
- python-casacore (MS extraction)
- pfb-imaging toolkit (provides `pfb hci` command)

## Quick Start

Generate 50 transients with SNR scaling and HCI injection:

```bash
# -o is the only required output argument
# Creates: run1_transients.yaml, run1.ecsv, run1.zarr, run1.fits, run1.reg
python transient_sim.py --ms /path/to/measurement.ms --nsources 50 -o run1
```

## Key Commands

### transient_sim.py - Main Orchestrator

```bash
# Default workflow (SNR scaling + HCI injection + FITS conversion)
python transient_sim.py --ms /path/to/ms --nsources 50 -o run1

# Custom SNR range and RMS
python transient_sim.py --ms /path/to/ms --nsources 50 \
    --snr-min 10 --snr-max 50 --rms 0.001 -o run1

# Skip FITS conversion (zarr only)
python transient_sim.py --ms /path/to/ms --nsources 50 -o run1 --no-fits

# Skip HCI injection (generate transients config only)
python transient_sim.py --ms /path/to/ms --nsources 50 -o run1 --no-hci
```

### verify_transients.py - Injection Verification

```bash
python verify_transients.py --manifest run1.ecsv --cube run1.zarr
```

### manifest_to_regions.py - DS9 Region Files

```bash
python manifest_to_regions.py --manifest manifest.ecsv -o transients.reg
```

## Output Formats

| Format | Description |
|--------|-------------|
| YAML | Transient definitions (input to HCI) |
| ECSV | Manifest with coordinates, fluxes, expected values |
| DS9 .reg | Region file for visualization |
| Zarr | Image cube from HCI injection |
| FITS | Converted cube for distribution |

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed documentation including:
- Module architecture and dependency graph
- Transient shape definitions and spectral correction
- SNR scaling methodology
- Pipeline flow diagrams
- Remote execution instructions
