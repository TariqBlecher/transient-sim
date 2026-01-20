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
    def from_ms(cls, ms_path: str, config: TransientConfig) -> "TransientSimulator":
        """Create simulator with parameters extracted from a Measurement Set."""
        return cls(ObservationParams.from_ms(ms_path), config)

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
    parser.add_argument("--output", "-o", required=True, help="Output YAML path")
    parser.add_argument(
        "--input-transients",
        "-i",
        help="Input transients YAML (skip generation, use for HCI)",
    )
    parser.add_argument("--manifest", "-m", help="Output manifest ECSV path")
    parser.add_argument(
        "--fov", type=float, default=2.0, help="Field of view (degrees)"
    )
    parser.add_argument(
        "--duration-max", type=float, default=10.0, help="Max duration (seconds)"
    )
    # SNR-based flux generation
    parser.add_argument(
        "--snr-min", type=float, default=5.0, help="Min target SNR (default: 5.0)"
    )
    parser.add_argument(
        "--snr-max", type=float, default=20.0, help="Max target SNR (default: 20.0)"
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
    parser.add_argument("--hci-output", help="Output directory for HCI zarr cube")
    parser.add_argument("--venv", help="Path to virtualenv (e.g. ~/venvs/sim_env)")
    parser.add_argument("--nx", type=int, default=3072, help="Image size x")
    parser.add_argument("--ny", type=int, default=3072, help="Image size y")
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
    args = parser.parse_args()

    # Check FOV against cube size
    max_fov_deg = min(args.nx, args.ny) * args.cell_size / 3600.0
    if args.fov > max_fov_deg:
        print(
            f"Warning: Requested FOV ({args.fov} deg) exceeds cube size ({max_fov_deg:.2f} deg)"
        )
        print(
            f'  Clamping FOV to {max_fov_deg:.2f} deg (cube: {args.nx}x{args.ny} @ {args.cell_size}")'
        )
        args.fov = max_fov_deg

    cfg = TransientConfig(
        fov_deg=args.fov,
        snr_range=(args.snr_min, args.snr_max),
        rms_jy=args.rms,
        duration_max_sec=args.duration_max,
    )
    sim = TransientSimulator.from_ms(args.ms, cfg)
    hci_cfg = HCIConfig(
        nx=args.nx,
        ny=args.ny,
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
        transient_yaml_path = Path(args.output)
        sim.save(args.output)

    # Run HCI with transients
    if args.run_hci:
        if not args.hci_output:
            args.hci_output = str(transient_yaml_path.with_suffix(".zarr"))
        print(f"Running HCI injection -> {args.hci_output}")
        result = run_hci(
            ms_path=sim.obs.ms_path,
            output_dir=args.hci_output,
            transient_yaml=str(transient_yaml_path),
            hci_config=hci_cfg,
            venv_path=args.venv,
        )
        if result["success"]:
            print(f"HCI complete in {result['elapsed_sec']:.1f}s")
            # Convert to FITS by default
            if not args.no_fits:
                # pfb hci creates zarr store at exact output path (no .zarr suffix)
                zarr_path = args.hci_output.rstrip("/")
                fits_path = zarr_path + ".fits"
                print(f"Converting to FITS -> {fits_path}")
                zarr_to_fits(zarr_path, fits_path)
        else:
            print(f"HCI failed: {result.get('stderr', 'unknown error')}")

    if args.manifest and sim.transients:
        sim.save_manifest(args.manifest)
        print(f"Manifest saved to {args.manifest}")
        # Auto-generate region file
        reg_path = str(Path(args.manifest).with_suffix(".reg"))
        count = manifest_to_regions(args.manifest, reg_path)
        print(f"Region file saved to {reg_path} ({count} regions)")
