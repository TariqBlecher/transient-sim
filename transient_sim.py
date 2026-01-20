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
from cube_utils import extract_rms_from_cube
from transient import Transient, TransientConfig, save_transients_yaml
from hci_runner import HCIConfig, run_hci
from zarr_to_fits_simple import zarr_to_fits


class TransientSimulator:
    """Generate synthetic transient configurations for radio transient simulations."""

    def __init__(self, obs_params: ObservationParams, config: Optional[TransientConfig] = None):
        self.obs = obs_params
        self.config = config or TransientConfig()
        self.transients: List[Transient] = []
        self.metadata: Dict[str, Any] = {}

    @classmethod
    def from_ms(cls, ms_path: str, config: Optional[TransientConfig] = None) -> 'TransientSimulator':
        """Create simulator with parameters extracted from a Measurement Set."""
        return cls(ObservationParams.from_ms(ms_path), config)

    def generate(self, nsources: int, seed: Optional[int] = None, name_prefix: str = 'sim') -> List[Transient]:
        """Generate transient source configurations."""
        rng = np.random.default_rng(seed)
        cfg = self.config
        obs = self.obs

        half_fov = cfg.fov_deg / 2.0
        duration_min = obs.integration_time_sec / 4  
        duration_max = min(cfg.duration_max_sec, obs.time_range_sec / 4)
        transients = []

        for i in range(nsources):
            # Generate DEC first
            dec = np.clip(obs.dec_center_deg + rng.uniform(-half_fov, half_fov), -90.0, 90.0)
            # Scale RA range by 1/cos(DEC) to account for spherical coordinates
            ra_half_fov = half_fov / np.cos(np.radians(dec))
            ra = (obs.ra_center_deg + rng.uniform(-ra_half_fov, ra_half_fov)) % 360.0
            peak_time = float(rng.uniform(0, obs.time_range_sec))
            duration = float(rng.uniform(duration_min, duration_max))

            periodic = rng.random() < cfg.periodic_fraction
            period = None
            if periodic:
                pmin = cfg.period_range_sec[0] if cfg.period_range_sec else duration
                pmax = cfg.period_range_sec[1] if cfg.period_range_sec else obs.time_range_sec * 0.3
                period = float(rng.uniform(pmin, pmax))

            shape = str(rng.choice(cfg.shapes))
            spectral_index = float(rng.uniform(*cfg.spectral_index_range))
            peak_flux = float(rng.uniform(*cfg.flux_range_jy))

            # Compute expected cube flux (assuming peak_time centered in bin)
            bin_start = peak_time - obs.integration_time_sec / 2
            bin_end = peak_time + obs.integration_time_sec / 2
            expected_flux = compute_expected_flux(
                peak_flux, peak_time, duration, shape, bin_start, bin_end,
                spectral_index=spectral_index,
                reference_freq=obs.reference_freq_hz,
                freq_min=obs.freq_min_hz,
                freq_max=obs.freq_max_hz
            )

            transients.append(Transient(
                name=f'{name_prefix}_{i:04d}',
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
                expected_cube_flux_jy=float(expected_flux)
            ))

        self.transients = transients
        self.metadata = {
            'generated_at': datetime.now().isoformat(),
            'nsources': nsources,
            'seed': seed,
            'ms_path': obs.ms_path,
            'time_range_sec': obs.time_range_sec,
            'fov_deg': cfg.fov_deg,
            'ra_center_deg': obs.ra_center_deg,
            'dec_center_deg': obs.dec_center_deg,
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
        shapes = {s: sum(1 for t in self.transients if t.shape == s) for s in set(t.shape for t in self.transients)}
        return (f"Generated {len(self.transients)} transients | "
                f"Flux: {min(fluxes):.2f}-{max(fluxes):.2f} Jy | Shapes: {shapes}")

    def scale_to_snr(
        self,
        snr_min: float,
        snr_max: float,
        rms: Optional[float] = None,
        seed: Optional[int] = None
    ) -> float:
        """
        Scale transient fluxes to ensure sensible, detectable values.

        This is an integral part of transient generation. Without scaling,
        randomly-generated fluxes from flux_range may produce transients that
        are either too faint to detect or unrealistically bright. Scaling
        ensures all transients fall within a target SNR range, making them
        suitable for testing detection algorithms.

        For each transient:
        - Compute expected_cube_flux from current peak_flux
        - Calculate SNR = expected_cube_flux / rms
        - If SNR outside range, scale peak_flux to hit random target in range

        Args:
            snr_min: Minimum target SNR
            snr_max: Maximum target SNR
            rms: RMS noise level in Jy (required)
            seed: Random seed for reproducibility

        Returns:
            RMS value used for scaling
        """
        if rms is None:
            raise ValueError("RMS must be provided. Generate a baseline cube first if needed.")
        if not self.transients:
            raise ValueError("No transients to scale. Call generate() first.")

        rng = np.random.default_rng(seed)
        obs = self.obs
        scaled_count = 0

        for t in self.transients:
            # Compute expected flux with current peak_flux
            bin_start = t.peak_time_sec - obs.integration_time_sec / 2
            bin_end = t.peak_time_sec + obs.integration_time_sec / 2

            expected_flux = compute_expected_flux(
                t.peak_flux_jy, t.peak_time_sec, t.duration_sec, t.shape, bin_start, bin_end,
                spectral_index=t.spectral_index,
                reference_freq=obs.reference_freq_hz,
                freq_min=obs.freq_min_hz,
                freq_max=obs.freq_max_hz
            )

            snr = expected_flux / rms

            # Scale if outside range
            if snr < snr_min or snr > snr_max:
                target_snr = float(rng.uniform(snr_min, snr_max))
                target_expected = target_snr * rms

                # Correction factor: expected_flux per unit peak_flux
                correction = compute_expected_flux(
                    1.0, t.peak_time_sec, t.duration_sec, t.shape, bin_start, bin_end,
                    spectral_index=t.spectral_index,
                    reference_freq=obs.reference_freq_hz,
                    freq_min=obs.freq_min_hz,
                    freq_max=obs.freq_max_hz
                )

                new_peak_flux = target_expected / correction if correction > 0 else target_expected
                t.peak_flux_jy = float(new_peak_flux)
                t.expected_cube_flux_jy = float(target_expected)
                scaled_count += 1
            else:
                t.expected_cube_flux_jy = float(expected_flux)

        # Update metadata
        self.metadata['rms_jy'] = rms
        self.metadata['snr_range'] = [snr_min, snr_max]
        self.metadata['scaled_count'] = scaled_count

        print(f"Scaled {scaled_count}/{len(self.transients)} transients to SNR range [{snr_min}, {snr_max}]")
        return rms

    def run_hci(
        self,
        output_dir: Union[str, Path],
        transient_yaml: Union[str, Path],
        hci_config: Optional[HCIConfig] = None,
        venv_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run pfb HCI injection with the generated transients.

        Args:
            output_dir: Output directory for zarr cube
            transient_yaml: Path to transient YAML file
            hci_config: HCI configuration (uses defaults if None)
            venv_path: Path to virtualenv to activate (e.g. ~/venvs/sim_env)

        Returns:
            Dict with runtime info and output path
        """
        result = run_hci(
            ms_path=self.obs.ms_path,
            output_dir=output_dir,
            transient_yaml=transient_yaml,
            hci_config=hci_config,
            venv_path=venv_path
        )
        # Update metadata
        self.metadata['hci_run'] = result
        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate transient simulations and run HCI injection')
    parser.add_argument('--ms', required=True, help='Path to Measurement Set')
    parser.add_argument('--nsources', type=int, default=10, help='Number of transients')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--output', '-o', required=True, help='Output YAML path')
    parser.add_argument('--input-transients', '-i', help='Input transients YAML (skip generation, use for HCI)')
    parser.add_argument('--manifest', '-m', help='Output manifest ECSV path')
    parser.add_argument('--fov', type=float, default=2.0, help='Field of view (degrees)')
    parser.add_argument('--flux-min', type=float, default=0.1, help='Min peak flux (Jy)')
    parser.add_argument('--flux-max', type=float, default=5.0, help='Max peak flux (Jy)')
    parser.add_argument('--duration-max', type=float, default=10.0, help='Max duration (seconds)')
    # SNR scaling options
    parser.add_argument('--scale-snr', action='store_true', dest='scale_snr', default=True,
                        help='Enable SNR-based flux scaling (default: enabled)')
    parser.add_argument('--no-scale-snr', action='store_false', dest='scale_snr',
                        help='Disable SNR-based flux scaling')
    parser.add_argument('--snr-min', type=float, default=5.0, help='Min target SNR (default: 5.0)')
    parser.add_argument('--snr-max', type=float, default=20.0, help='Max target SNR (default: 20.0)')
    parser.add_argument('--rms', type=float, default=1.4e-04, help='RMS noise in Jy (default: 1.4e-04 from SM1R00C04_1min baseline)')
    # HCI options
    parser.add_argument('--run-hci', action='store_true', help='Run HCI injection after generating transients')
    parser.add_argument('--hci-output', help='Output directory for HCI zarr cube')
    parser.add_argument('--venv', help='Path to virtualenv (e.g. ~/venvs/sim_env)')
    parser.add_argument('--nx', type=int, default=3072, help='Image size x')
    parser.add_argument('--ny', type=int, default=3072, help='Image size y')
    parser.add_argument('--cell-size', type=float, default=2.4, help='Cell size (arcsec)')
    parser.add_argument('--nworkers', type=int, default=16, help='Number of workers')
    # FITS output options
    parser.add_argument('--no-fits', action='store_true', help='Skip zarr to FITS conversion (default: convert)')
    args = parser.parse_args()

    # Check FOV against cube size
    max_fov_deg = min(args.nx, args.ny) * args.cell_size / 3600.0
    if args.fov > max_fov_deg:
        print(f"Warning: Requested FOV ({args.fov} deg) exceeds cube size ({max_fov_deg:.2f} deg)")
        print(f"  Clamping FOV to {max_fov_deg:.2f} deg (cube: {args.nx}x{args.ny} @ {args.cell_size}\")")
        args.fov = max_fov_deg

    # SNR scaling enabled by default, can be disabled with --no-scale-snr
    snr_scaling_enabled = args.scale_snr

    cfg = TransientConfig(
        fov_deg=args.fov,
        flux_range_jy=(args.flux_min, args.flux_max),
        duration_max_sec=args.duration_max
    )
    sim = TransientSimulator.from_ms(args.ms, cfg)
    hci_cfg = HCIConfig(nx=args.nx, ny=args.ny, cell_size=args.cell_size,
                        fov=args.fov, nworkers=args.nworkers)

    # If input transients provided, skip generation and use existing file for HCI
    if args.input_transients:
        input_yaml = Path(args.input_transients)
        if not input_yaml.exists():
            print(f"Error: Input transients file not found: {input_yaml}")
            exit(1)
        print(f"Using existing transients from {input_yaml}")
        transient_yaml_path = input_yaml
    else:
        sim.generate(nsources=args.nsources, seed=args.seed)
        print(sim.summary())
        transient_yaml_path = Path(args.output)

        # SNR Scaling workflow
        if snr_scaling_enabled:
            rms = args.rms

            # If no RMS provided, generate baseline cube to measure noise
            if rms is None:
                if not args.hci_output:
                    args.hci_output = str(transient_yaml_path.with_suffix('.zarr'))
                baseline_dir = Path(args.hci_output).parent / 'baseline.zarr'
                print(f"Generating baseline cube (no transients) -> {baseline_dir}")

                baseline_result = run_hci(
                    ms_path=sim.obs.ms_path,
                    output_dir=baseline_dir,
                    transient_yaml=None,  # No transients for baseline
                    hci_config=hci_cfg,
                    venv_path=args.venv
                )

                if not baseline_result['success']:
                    print(f"Baseline HCI failed: {baseline_result.get('stderr', 'unknown error')}")
                    exit(1)

                print(f"Baseline cube complete in {baseline_result['elapsed_sec']:.1f}s")
                rms = extract_rms_from_cube(baseline_dir)
                print(f"Extracted RMS from baseline: {rms:.4f} Jy")

            # Scale transients to target SNR range
            sim.scale_to_snr(args.snr_min, args.snr_max, rms=rms, seed=args.seed)

        # Save transients (potentially scaled)
        sim.save(args.output)

    # Run HCI with (potentially scaled) transients
    if args.run_hci:
        if not args.hci_output:
            args.hci_output = str(transient_yaml_path.with_suffix('.zarr'))
        print(f"Running HCI injection -> {args.hci_output}")
        result = sim.run_hci(args.hci_output, str(transient_yaml_path), hci_cfg, args.venv)
        if result['success']:
            print(f"HCI complete in {result['elapsed_sec']:.1f}s")
            # Convert to FITS by default
            if not args.no_fits:
                fits_path = args.hci_output.rstrip('/').replace('.zarr', '.fits')
                print(f"Converting to FITS -> {fits_path}")
                zarr_to_fits(args.hci_output, fits_path)
        else:
            print(f"HCI failed: {result.get('stderr', 'unknown error')}")

    if args.manifest and sim.transients:
        sim.save_manifest(args.manifest)
        print(f"Manifest saved to {args.manifest}")
        # Auto-generate region file
        reg_path = str(Path(args.manifest).with_suffix('.reg'))
        count = manifest_to_regions(args.manifest, reg_path)
        print(f"Region file saved to {reg_path} ({count} regions)")
