# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style Preferences

- **Prefer shorter, simpler code over features/complexity** - Keep implementations minimal and straightforward

## Overview

Radio transient source simulation framework for astronomical research. Generates synthetic transients and injects them into radio interferometry data cubes using pfb-imaging for validation and testing of transient detection algorithms.

## Key Commands

### Full workflow: Generate, scale, inject, verify

```bash
# 1. Generate transients with initial flux range
python transient_sim.py --ms /path/to/ms --nsources 50 --seed 42 \
    -o transients.yaml -m manifest.ecsv \
    --flux-min 0.1 --flux-max 5.0 --duration-max 180

# 2. Scale fluxes to achieve 5-20 sigma in output cube
python scale_fluxes.py --manifest manifest.ecsv --rms-cube baseline.zarr \
    --ms /path/to/ms --snr-min 5 --snr-max 20 \
    -o scaled.yaml -m scaled_manifest.ecsv

# 3. Run HCI injection with scaled transients
python transient_sim.py --ms /path/to/ms -o scaled.yaml \
    --run-hci --hci-output output_cube --venv ~/venvs/sim_env

# 4. Verify injection
python verify_transients.py --manifest scaled_manifest.ecsv --cube output_cube \
    --ms /path/to/ms
```

### Compare cubes with/without transients
```bash
python compare_cubes.py --baseline baseline_cube --injected injected_cube \
    --manifest manifest.ecsv
```

### Generate DS9 region file for visualization
```bash
python manifest_to_regions.py --manifest manifest.ecsv [-o output.reg] [--radius 10]
```

## Architecture

### Core Module: transient_sim.py
- `ObservationParams.from_ms()` - Extracts time range, frequency, RA/Dec from Measurement Set using python-casacore
- `TransientSimulator` - Main class: `generate()` → `save()` → `run_hci()`
- `compute_expected_flux()` - Computes expected measured flux accounting for shape, time-binning, and spectral index
- `compute_spectral_correction()` - Band-averaged flux correction for spectral index
- Duration range: `[integration_time, duration_max_sec]` (default max 180s)
- Outputs YAML for pfb and ECSV manifest (astropy table format) with metadata in header

### Flux Scaling: scale_fluxes.py
- Reads manifest and baseline cube RMS
- For each transient: if expected SNR < 5 or > 20, scales `peak_flux_jy` to hit random target in range
- Outputs updated YAML and manifest with `expected_cube_flux_jy` field

### I/O Module: manifest_io.py
- `save_manifest_ecsv()` - Saves transient list to ECSV with metadata
- `load_manifest_ecsv()` - Returns (DataFrame, metadata dict)

### Verification: verify_transients.py
- Compares expected vs measured flux in output cubes
- Uses same flux computation logic as transient_sim.py

## Transient Shape Conventions (pfb-imaging)

These match pfb's implementation in `pfb/utils/transients.py`:

| Shape | Parameter Meaning | Profile |
|-------|-------------------|---------|
| gaussian | σ = duration (not FWHM) | `exp(-0.5 * ((t - peak_time) / duration)^2)` |
| exponential | τ = duration | `exp(-(t - peak_time) / duration)` for t > peak_time |
| step | duration = active period | Constant flux from peak_time to peak_time+duration |

## Flux Computation

The `peak_flux_jy` in simulations is the **peak of the shape function**, but the **measured flux in cubes is lower** due to:

1. **Time integration**: Cube has 8s time bins; shape function gets averaged over each bin:
   - `step`: If duration ≥ bin width → measured ≈ peak_flux
   - `gaussian/exponential`: Peak gets smeared → measured < peak_flux

2. **Spectral index**: `flux(ν) = peak_flux * (ν/ν_ref)^α` averaged over band (0.856-1.711 GHz). Negative α (typical) → band-averaged flux < peak_flux at ν_ref.

3. **Bin alignment**: If peak_time falls between bin centers, flux is split across bins.

**Important**: SNR estimates using `peak_flux / RMS` are optimistic. Actual measured SNR will be lower depending on shape, duration, and spectral index. See `compute_expected_flux()` in `transient_sim.py` for the full calculation. Use `scale_fluxes.py` to ensure transients fall within a target SNR range (default 5-20 sigma).

**Known issue**: Flux prediction accuracy varies by shape:
- `gaussian`: Accurate (measured/expected ratio ~1.0)
- `exponential`: Slightly high (~1.2x expected)
- `step`: Systematic over-measurement (~2x expected) - likely time-bin alignment issue in `compute_expected_flux()`

## pfb hci Notes

**IMPORTANT**: Do not run `pfb hci` directly unless testing something different from the normal workflow. Use `transient_sim.py --run-hci` which has the correct parameters built in.

If you must run `pfb hci` directly, use ALL required parameters:

```bash
pfb hci \
  --ms /vault2-tina/blecher/SMC/transient_sim/SM1R00C04_1min.ms \
  -o /vault2-tina/blecher/SMC/transient_sim/pfb_approach/hci_inject \
  --output-format zarr \
  --data-column CORRECTED_DATA-MODEL_DATA \
  --weight-column WEIGHT_SPECTRUM \
  --nx 3072 --ny 3072 \
  --cell-size 2.4 \
  -fov 2.0 \
  --product I \
  --nworkers 16 \
  --overwrite \
  --robustness 0 \
  --stack \
  --inject-transients transients.yaml
```

**Critical parameters** (omitting these causes ~8x higher RMS noise):
- `--data-column CORRECTED_DATA-MODEL_DATA` - Use calibrated residual data, NOT raw `DATA`
- `--weight-column WEIGHT_SPECTRUM` - Use proper visibility weights
- `--robustness 0` - Briggs weighting
- `--stack` - Create single stacked cube (without this, pfb creates individual zarr files per time step)

## Utilities

### zarr_to_fits_simple.py
Converts zarr cubes to FITS format for distribution:
```bash
python zarr_to_fits_simple.py input.zarr output.fits
```
Uses streaming HDU writer for memory efficiency with large cubes.

## Dependencies

- numpy, scipy, xarray (zarr), astropy, pyyaml
- python-casacore (MS extraction)
- pfb-imaging toolkit (external, provides `pfb hci` command)

## Remote Execution (tina)

SSH: `blecher@tina.ru.ac.za`

**Running commands on tina**:
```bash
ssh blecher@tina.ru.ac.za ". ~/venvs/sim_env/bin/activate && python3 script.py"
```

### Key Paths

| Path | Description |
|------|-------------|
| `/vault2-tina/blecher/SMC/transient_sim/SM1R00C04_1min.ms` | Measurement Set (SMC field, 1-min integration) |
| `/vault2-tina/blecher/SMC/transient_sim/pfb_approach/` | Output cubes directory |
| `/vault2-tina/blecher/smc_notebooks/transient_sims/` | Scripts and notebooks |
| `~/venvs/sim_env` | Python virtualenv with pfb, casacore, etc. |

### Scripts on tina (`transient_sims/`)

| File | Purpose |
|------|---------|
| `transient_sim.py` | Generates transients, computes expected flux, runs HCI |
| `scale_fluxes.py` | Scales peak fluxes to achieve target SNR range (5-20 sigma) |
| `manifest_io.py` | ECSV manifest I/O functions |
| `verify_transients.py` | Verifies injected transients match expected flux |
| `compare_cubes.py` | Compares baseline vs injected cubes |
| `manifest_to_regions.py` | Converts manifest ECSV to DS9 region file for CARTA/DS9 visualization |
| `verify_simple.py` | Simple verification for controlled tests |
| `create_transient_yaml.ipynb` | Original notebook for generating transient configs |
| `inspect_transients.ipynb` | Visualize injected transients in cubes |

### Test Artifacts on tina

| File/Dir | Description |
|----------|-------------|
| `test_pipeline.yaml` + `_manifest.json` | 50 transients, varied shapes/spectral indices |
| `test_simple.yaml` | 20 transients (step, duration=8s, α=0) |
| `test_100.yaml` + `_manifest.json` | 100 transients test run |

### Output Cubes (`pfb_approach/`)

| Cube | Description |
|------|-------------|
| `hci_no_transients` | Baseline cube (no injection) |
| `hci_pipeline_test` | 50 transients with varied shapes/spectral indices |
| `hci_simple_test` | 20 step transients (duration=8s, α=0) - **20/20 verified** |
| `hci_inject_100` | 100 transients test |
| `hci_inject_single` | Single transient test |
| `hci_inject_flat.zarr` | Flat spectrum test (α=0) |

### Deprecated (`Depreciated/`)

Old scripts and notebooks no longer maintained:
- `debug_transients.py` - Debug failing transients with lightcurve plots
- `debug_step.py` - Debug step-shaped transients using difference cubes
- `compare transient results.ipynb` - manual comparison (now `compare_cubes.py`)
- `Simulated Transients into cube simple.ipynb` - early prototype

## Task Memory System

Ephemeral task context is auto-saved to `.claude/tasks/` on session end and context compaction. These files capture:
- Files modified and read (names only)
- Commands run
- Placeholders for notes and next steps

**When resuming work**: Read only the most recent task file(s) first (sorted by date). Don't read all task files upfront - they're for context on specific in-progress work, not general project knowledge.

Task files are named `YYYY-MM-DD_<slug>.md` where slug comes from the first prompt.

## Maintaining This File

Run `/update-docs` to update this file with new learnings from the current session.

## Files for the team

 When creating datasets for distribution for the broader team, it's important to keep names of fields and observations and not use generic names like measurement_set.ms instead use e.g. 1564606566_sdp_l0_1024SMC1R04C04_sim.ms
