"""
Transient Simulation Generator

Generates synthetic transient source configurations for injection into radio data cubes.
Auto-extracts observation parameters from Measurement Sets using python-casacore.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from datetime import datetime

from manifest_to_regions import manifest_to_regions
from flux_utils import compute_expected_flux
from obs_params import ObservationParams
from transient import Transient, TransientConfig, save_transients_yaml
from hci_runner import HCIConfig, run_hci
from zarr_to_fits_simple import zarr_to_fits


class TransientSimulator:
    """Generate synthetic transient configurations for radio transient simulations."""

    def __init__(self, obs_params: ObservationParams, config: TransientConfig):
        self.obs = obs_params
        self.config = config
        self.transients: List[Transient] = []
        self.metadata: Dict[str, Any] = {}

    @classmethod
    def from_ms(
        cls, ms_path: str, config: TransientConfig, extract_scans: bool = True
    ) -> "TransientSimulator":
        """Create simulator with parameters extracted from a Measurement Set.

        Args:
            ms_path: Path to the Measurement Set
            config: TransientConfig with generation parameters
            extract_scans: If True, extract scan boundaries for scan-aware peak time
                sampling. Set to False to sample from the entire observation.
        """
        return cls(ObservationParams.from_ms(ms_path, extract_scans=extract_scans), config)

    def generate(
        self, nsources: int, seed: Optional[int] = None, name_prefix: str = "sim"
    ) -> List[Transient]:
        """Generate transient source configurations."""
        rng = np.random.default_rng(seed)
        cfg = self.config
        obs = self.obs

        half_fov = cfg.fov_deg / 2.0
        duration_min = obs.integration_time_sec
        duration_max = min(cfg.duration_max_sec, obs.time_range_sec / 4)
        transients = []

        for i in range(nsources):
            # Generate DEC first
            dec = np.clip(
                obs.dec_center_deg + rng.uniform(-half_fov, half_fov), -90.0, 90.0
            )
            # Scale RA range by 1/cos(DEC) to account for spherical coordinates
            ra_half_fov = half_fov / np.cos(np.radians(dec))
            ra = (obs.ra_center_deg + rng.uniform(-ra_half_fov, ra_half_fov)) % 360.0
            # Sample peak_time within science scans only (if scan info available)
            if obs.scans:
                peak_time = obs.sample_science_time(rng)
            else:
                peak_time = float(rng.uniform(0, obs.time_range_sec))
            duration = float(rng.uniform(duration_min, duration_max))

            periodic = rng.random() < cfg.periodic_fraction
            period = None
            if periodic:
                pmin = cfg.period_range_sec[0] if cfg.period_range_sec else duration
                pmax = (
                    cfg.period_range_sec[1]
                    if cfg.period_range_sec
                    else obs.time_range_sec * 0.3
                )
                period = float(rng.uniform(pmin, pmax))

            shape = str(rng.choice(cfg.shapes))
            spectral_index = float(rng.uniform(*cfg.spectral_index_range))
            # Direct SNR-based flux: peak_flux = target_snr * rms
            target_snr = float(rng.uniform(*cfg.snr_range))
            peak_flux = target_snr * cfg.rms_jy

            # Compute expected cube flux (assuming peak_time centered in bin)
            bin_start = peak_time - obs.integration_time_sec / 2
            bin_end = peak_time + obs.integration_time_sec / 2
            expected_flux = compute_expected_flux(
                peak_flux,
                peak_time,
                duration,
                shape,
                bin_start,
                bin_end,
                spectral_index=spectral_index,
                reference_freq=obs.reference_freq_hz,
                freq_min=obs.freq_min_hz,
                freq_max=obs.freq_max_hz,
            )

            transients.append(
                Transient(
                    name=f"{name_prefix}_{i:04d}",
                    ra_deg=float(ra),
                    dec_deg=float(dec),
                    peak_time_sec=peak_time,
                    duration_sec=float(duration),
                    shape=shape,
                    peak_flux_jy=float(peak_flux),
                    spectral_index=spectral_index,
                    reference_freq_hz=obs.reference_freq_hz,
                    periodic=periodic,
                    period_sec=period,
                    expected_cube_flux_jy=float(expected_flux),
                )
            )

        self.transients = transients
        self.metadata = {
            "generated_at": datetime.now().isoformat(),
            "nsources": nsources,
            "seed": seed,
            "ms_path": obs.ms_path,
            "time_range_sec": obs.time_range_sec,
            "fov_deg": cfg.fov_deg,
            "ra_center_deg": obs.ra_center_deg,
            "dec_center_deg": obs.dec_center_deg,
            "snr_range": list(cfg.snr_range),
            "rms_jy": cfg.rms_jy,
        }
        return transients

    def save(self, output_path: Union[str, Path]) -> Path:
        """Save transients to YAML file."""
        return save_transients_yaml(self.transients, output_path)

    def save_manifest(self, output_path: Union[str, Path]) -> Path:
        """Save generation manifest with metadata and full transient details to ECSV."""
        from manifest_io import save_manifest_ecsv

        return save_manifest_ecsv(self.transients, output_path, self.metadata)

    def summary(self) -> str:
        """Return a summary string."""
        if not self.transients:
            return "No transients generated."
        fluxes = [t.peak_flux_jy for t in self.transients]
        shapes = {
            s: sum(1 for t in self.transients if t.shape == s)
            for s in set(t.shape for t in self.transients)
        }
        cfg = self.config
        return (
            f"Generated {len(self.transients)} transients | "
            f"Flux: {min(fluxes):.4f}-{max(fluxes):.4f} Jy | "
            f"SNR range: {cfg.snr_range[0]}-{cfg.snr_range[1]} | Shapes: {shapes}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate transient simulations and run HCI injection"
    )
    parser.add_argument("--ms", required=True, help="Path to Measurement Set")
    parser.add_argument("--nsources", type=int, default=10, help="Number of transients")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--output", "-o", required=True,
        help="Base name for outputs (derives .yaml, .ecsv, .zarr, .fits, .reg)"
    )
    parser.add_argument(
        "--input-transients",
        "-i",
        help="Input transients YAML (skip generation, use for HCI)",
    )
    parser.add_argument(
        "--fov", type=float, default=2.0, help="Field of view (degrees)"
    )
    parser.add_argument(
        "--duration-max", type=float, default=10.0, help="Max duration (seconds)"
    )
    # SNR-based flux generation
    parser.add_argument(
        "--snr-min", type=float, default=8.0, help="Min target SNR (default: 8.0)"
    )
    parser.add_argument(
        "--snr-max", type=float, default=25.0, help="Max target SNR (default: 25.0)"
    )
    parser.add_argument(
        "--rms",
        type=float,
        default=1.4e-04,
        help="RMS noise in Jy (default: 1.4e-04 from SM1R00C04_1min baseline)",
    )
    # HCI options
    parser.add_argument(
        "--run-hci",
        action="store_true",
        help="Run HCI injection after generating transients",
    )
    parser.add_argument("--venv", help="Path to virtualenv (e.g. ~/venvs/sim_env)")
    parser.add_argument("--npix", type=int, default=3072, help="Image size (pixels)")
    parser.add_argument(
        "--cell-size", type=float, default=2.4, help="Cell size (arcsec)"
    )
    parser.add_argument("--nworkers", type=int, default=16, help="Number of workers")
    # FITS output options
    parser.add_argument(
        "--no-fits",
        action="store_true",
        help="Skip zarr to FITS conversion (default: convert)",
    )
    # Scan-aware sampling
    parser.add_argument(
        "--no-scan-aware",
        action="store_true",
        help="Disable scan-aware peak time sampling (sample from entire observation)",
    )
    args = parser.parse_args()

    # Derive all output paths from -o
    output_path = Path(args.output)
    base = output_path.with_suffix("")  # Remove any extension (handles both "run1" and "run1.yaml")
    # Use _transients suffix for YAML to avoid collision with pfb's intermediate zarr
    # (pfb creates {yaml_basename}.zarr for transient light curves)
    yaml_path = base.parent / f"{base.name}_transients.yaml"
    manifest_path = base.with_suffix(".ecsv")
    zarr_path = base.with_suffix(".zarr")

    # Check FOV against cube size
    max_fov_deg = args.npix * args.cell_size / 3600.0
    if args.fov > max_fov_deg:
        print(
            f"Warning: Requested FOV ({args.fov} deg) exceeds cube size ({max_fov_deg:.2f} deg)"
        )
        print(
            f'  Clamping FOV to {max_fov_deg:.2f} deg (cube: {args.npix}x{args.npix} @ {args.cell_size}")'
        )
        args.fov = max_fov_deg

    cfg = TransientConfig(
        fov_deg=args.fov,
        snr_range=(args.snr_min, args.snr_max),
        rms_jy=args.rms,
        duration_max_sec=args.duration_max,
    )
    extract_scans = not args.no_scan_aware
    sim = TransientSimulator.from_ms(args.ms, cfg, extract_scans=extract_scans)

    # Report scan extraction status
    if sim.obs.scans:
        science_scans = sim.obs.get_science_scans()
        total_science_time = sum(s.duration_sec for s in science_scans)
        print(
            f"Scan-aware sampling: {len(science_scans)} science scans, "
            f"{total_science_time:.1f}s total ({total_science_time/sim.obs.time_range_sec*100:.1f}% of observation)"
        )
    elif extract_scans:
        print("Warning: No scan info found, using full time range")
    else:
        print("Scan-aware sampling disabled, using full time range")

    hci_cfg = HCIConfig(
        nx=args.npix,
        ny=args.npix,
        cell_size=args.cell_size,
        fov=args.fov,
        nworkers=args.nworkers,
    )

    # If input transients provided, skip generation and use existing file for HCI
    if args.input_transients:
        input_yaml = Path(args.input_transients)
        if not input_yaml.exists():
            print(f"Error: Input transients file not found: {input_yaml}")
            exit(1)
        print(f"Using existing transients from {input_yaml}")
        transient_yaml_path = input_yaml
    else:
        # Generate transients with direct SNR-based flux
        sim.generate(nsources=args.nsources, seed=args.seed)
        print(sim.summary())
        transient_yaml_path = yaml_path
        sim.save(yaml_path)

    # Run HCI with transients
    if args.run_hci:
        print(f"Running HCI injection -> {zarr_path}")
        result = run_hci(
            ms_path=sim.obs.ms_path,
            output_dir=str(zarr_path),
            transient_yaml=str(transient_yaml_path),
            hci_config=hci_cfg,
            venv_path=args.venv,
        )
        if result["success"]:
            print(f"HCI complete in {result['elapsed_sec']:.1f}s")
            # Convert to FITS by default
            if not args.no_fits:
                fits_path = zarr_path.with_suffix(".fits")
                print(f"Converting to FITS -> {fits_path}")
                zarr_to_fits(str(zarr_path), str(fits_path))
        else:
            print(f"HCI failed: {result.get('stderr', 'unknown error')}")

    # Always save manifest when transients were generated (not when using --input-transients)
    if sim.transients:
        sim.save_manifest(manifest_path)
        print(f"Manifest saved to {manifest_path}")
        # Auto-generate region file
        reg_path = manifest_path.with_suffix(".reg")
        count = manifest_to_regions(str(manifest_path), str(reg_path))
        print(f"Region file saved to {reg_path} ({count} regions)")
