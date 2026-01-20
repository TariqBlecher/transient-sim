# Radio Transient Simulation Framework

A modular framework for generating and verifying synthetic radio transients for injection into visibility data.

## Module Architecture

```
sims/
├── flux_utils.py        # Pure flux computation (no external deps beyond numpy/scipy)
├── obs_params.py        # MS parameter extraction (casacore)
├── transient.py         # Transient dataclass + YAML I/O
├── cube_utils.py        # Zarr cube I/O + WCS utilities
├── hci_runner.py        # HCI subprocess orchestration
├── manifest_io.py       # ECSV manifest I/O
├── transient_sim.py     # CLI + TransientSimulator facade (MASTER ORCHESTRATOR)
├── scale_fluxes.py      # Ensures sensible fluxes (integrated into transient_sim, also standalone CLI)
├── verify_transients.py # Injection verification CLI
├── manifest_to_regions.py # DS9 region file generation
└── Depreciated/         # Archived scripts
```

## Dependency Graph

```
flux_utils.py (leaf - no internal deps)
    └── numpy, scipy

obs_params.py (leaf - no internal deps)
    └── numpy, casacore

transient.py
    └── (no internal deps, pure dataclasses + YAML)

cube_utils.py
    └── numpy, xarray, astropy

hci_runner.py
    └── subprocess, pathlib

manifest_io.py
    └── pandas, astropy

transient_sim.py (MASTER ORCHESTRATOR)
    ├── flux_utils
    ├── obs_params
    ├── cube_utils
    ├── transient
    ├── hci_runner
    ├── manifest_io
    ├── scale_fluxes  ← integrated, not separate step
    └── zarr_to_fits_simple  ← FITS conversion (default)

scale_fluxes.py (integrated into transient_sim; also available as standalone CLI)
    ├── flux_utils
    ├── obs_params
    ├── cube_utils
    ├── transient
    └── manifest_io

verify_transients.py
    ├── flux_utils
    ├── obs_params
    ├── cube_utils
    └── manifest_io
```

## Key Concepts

### Transient Shapes
- **gaussian**: `sigma = duration`, NOT FWHM-based
- **exponential**: `tau = duration`, starts at `peak_time`
- **step**: constant flux from `peak_time` to `peak_time + duration`

### Spectral Correction
Power-law spectrum: `flux(v) = peak_flux * (v/v_ref)^alpha`

The cube averages over frequency, so we compute the band-averaged flux correction factor.

### Time Bin Integration
Each time bin is centered on the timestamp with width = integration time. Expected flux accounts for the temporal profile overlap with the bin.

### SNR Scaling (Ensuring Sensible Fluxes)
SNR scaling is an **integral part of transient generation**, not a separate post-processing step. Without scaling, randomly-generated fluxes from `flux_range` may produce transients that are either too faint to detect or unrealistically bright compared to the noise level.

**Why this matters:** If you're testing a detection algorithm, you need transients with realistic, detectable fluxes. Scaling ensures all transients fall within a controlled SNR range, making simulations meaningful for algorithm validation.

- **Enabled by default** (`--scale-snr`), disable with `--no-scale-snr`
- Default SNR range: 5-20 (configurable via `--snr-min`, `--snr-max`)
- Default RMS: 1.4e-04 Jy (from SM1R00C04_1min baseline, override with `--rms`)
- Scale `peak_flux` so expected cube flux falls within `[snr_min, snr_max] * rms`

## CLI Usage

### Recommended Workflow (Integrated)

**Generate transients with SNR scaling (default) and HCI injection:**
```bash
# Uses default RMS (1.4e-04) and SNR range [5, 20]
python transient_sim.py --ms /path/to/ms --nsources 50 \
    --run-hci --hci-output injected.zarr \
    -o transients.yaml -m manifest.ecsv

# Custom SNR range and RMS
python transient_sim.py --ms /path/to/ms --nsources 50 \
    --snr-min 10 --snr-max 50 --rms 0.001 \
    --run-hci --hci-output injected.zarr \
    -o transients.yaml -m manifest.ecsv

# Skip FITS conversion (zarr only)
python transient_sim.py --ms /path/to/ms --nsources 50 \
    --run-hci --hci-output injected.zarr --no-fits \
    -o transients.yaml -m manifest.ecsv
```

**Generate transients WITHOUT scaling (use raw flux_range):**
```bash
python transient_sim.py --ms /path/to/ms --nsources 50 \
    --no-scale-snr \
    -o transients.yaml -m manifest.ecsv
```

### Standalone Scale Fluxes (Special Cases)

Normally, scaling is handled automatically by `transient_sim.py`. The standalone CLI is for re-scaling an existing manifest (e.g., to test different SNR ranges without regenerating transients):
```bash
# Re-scale existing manifest with known RMS
python scale_fluxes.py --manifest manifest.ecsv --rms 0.001 \
    --ms /path/to/ms -o scaled.yaml --snr-min 10 --snr-max 30

# Or extract RMS from a cube
python scale_fluxes.py --manifest manifest.ecsv --rms-cube baseline.zarr \
    --ms /path/to/ms -o scaled.yaml --snr-min 5 --snr-max 20
```

### Verify Injection
```bash
python verify_transients.py --manifest manifest.ecsv --cube output.zarr --ms /path/to/ms
```

## Pipeline Flow

```
transient_sim.py --ms ... --run-hci [--rms 0.001]
    │
    ├─ Step 1: Generate initial transients (random fluxes in flux_range)
    │
    ├─ Step 2: SNR Scaling (DEFAULT ON, disable with --no-scale-snr)
    │    ├─ If --rms provided → use directly
    │    └─ Else → run HCI (no transients) → baseline.zarr → extract RMS
    │
    ├─ Step 3: Scale peak_flux to hit target SNR range [5, 20] by default
    │
    ├─ Step 4: Save scaled YAML + manifest
    │
    ├─ Step 5: Run HCI with scaled transients → final.zarr
    │
    └─ Step 6: Convert to FITS (DEFAULT ON, disable with --no-fits) → final.fits
```

## Development Notes

- **Prefer shorter, simpler code over features/complexity** - Keep implementations minimal and straightforward
- All flux calculations go through `flux_utils.py` - single source of truth
- `ObservationParams.from_ms()` handles all MS extraction
- `Transient` dataclass is the canonical representation with factory methods for different sources
- `Transient.from_manifest_row()` creates instances from manifest DataFrame rows
- Lazy imports for heavy dependencies (casacore, xarray) where appropriate
- `transient_sim.py` is the master orchestrator - prefer using it over separate CLI steps
- `run_hci()` accepts `transient_yaml=None` to generate baseline cubes without transients

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

## Task Memory System

Ephemeral task context is auto-saved to `.claude/tasks/` on session end and context compaction. These files capture:
- Files modified and read (names only)
- Commands run
- Placeholders for notes and next steps

**When resuming work**: Read only the most recent task file(s) first (sorted by date). Don't read all task files upfront - they're for context on specific in-progress work, not general project knowledge.

Task files are named `YYYY-MM-DD_<slug>.md` where slug comes from the first prompt.

## Files for the Team

When creating datasets for distribution for the broader team, it's important to keep names of fields and observations and not use generic names like `measurement_set.ms` — instead use e.g. `1564606566_sdp_l0_1024SMC1R04C04_sim.ms`.

## Maintaining This File

Run `/update-docs` to update this file with new learnings from the current session.
