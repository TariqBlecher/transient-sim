"""
Transient Simulation Generator

Generates synthetic transient source configurations for injection into radio data cubes.
Auto-extracts observation parameters from Measurement Sets using python-casacore.
"""

import yaml
import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict, Any
from datetime import datetime
from math import floor
from scipy import integrate
from manifest_to_regions import manifest_to_regions


def extract_rms_from_cube(cube_path):
    """Extract mean RMS from a zarr cube."""
    import xarray as xr
    ds = xr.open_zarr(cube_path)
    return float(ds.rms.to_numpy().mean())


def compute_spectral_correction(spectral_index, reference_freq, freq_min, freq_max):
    """
    Compute the flux correction factor for spectral index.

    pfb applies: fprofile(ν) = peak_flux * (ν/ν_ref)^α
    The cube averages over frequency, so effective flux = mean(fprofile) over band.

    Returns the ratio of band-averaged flux to peak_flux.
    """
    alpha = spectral_index
    if abs(alpha) < 1e-10:
        return 1.0
    if abs(alpha + 1) < 1e-10:
        return reference_freq * np.log(freq_max / freq_min) / (freq_max - freq_min)
    integral = (freq_max**(alpha + 1) - freq_min**(alpha + 1)) / (alpha + 1)
    average = integral / (freq_max - freq_min)
    return average / (reference_freq ** alpha)


def compute_expected_flux(peak_flux, peak_time, duration, shape, bin_start, bin_end,
                          spectral_index=0.0, reference_freq=None, freq_min=None, freq_max=None):
    """
    Compute expected measured flux in a time bin, accounting for shape and spectral index.

    Matches pfb-imaging implementation in pfb/utils/transients.py:
    - gaussian: sigma = duration (NOT FWHM-based)
    - exponential: tau = duration
    - step: active for [peak_time, peak_time + duration]
    """
    bin_width = bin_end - bin_start
    if reference_freq is not None and freq_min is not None and freq_max is not None:
        spec_correction = compute_spectral_correction(spectral_index, reference_freq, freq_min, freq_max)
    else:
        spec_correction = 1.0
    effective_peak_flux = peak_flux * spec_correction

    if shape == "step":
        t_start = max(bin_start, peak_time)
        t_end = min(bin_end, peak_time + duration)
        overlap = max(0, t_end - t_start)
        return effective_peak_flux * overlap / bin_width

    elif shape == "gaussian":
        sigma = duration
        def gaussian(t):
            return effective_peak_flux * np.exp(-((t - peak_time) ** 2) / (2 * sigma ** 2))
        integral, _ = integrate.quad(gaussian, bin_start, bin_end)
        return integral / bin_width

    elif shape == "exponential":
        tau = duration
        if bin_end <= peak_time:
            return 0.0
        t_start = max(bin_start, peak_time)
        def exponential(t):
            return effective_peak_flux * np.exp(-(t - peak_time) / tau)
        integral, _ = integrate.quad(exponential, t_start, bin_end)
        return integral / bin_width

    return 0.0


@dataclass
class ObservationParams:
    """Parameters extracted from a Measurement Set."""
    ms_path: str
    time_range_sec: float
    n_timesteps: int
    integration_time_sec: float
    ra_center_deg: float
    dec_center_deg: float
    reference_freq_hz: float
    freq_min_hz: float
    freq_max_hz: float

    @classmethod
    def from_ms(cls, ms_path: str) -> 'ObservationParams':
        """Extract observation parameters from a Measurement Set."""
        from casacore.tables import table

        ms_path = str(ms_path)

        with table(ms_path, readonly=True, ack=False) as t:
            times = t.getcol('TIME')
            unique_times = np.unique(times)

        time_range = times.max() - times.min()
        n_timesteps = len(unique_times)
        integration_time = float(np.median(np.diff(unique_times))) if n_timesteps > 1 else time_range

        with table(f'{ms_path}/FIELD', readonly=True, ack=False) as field_table:
            phase_dir = field_table.getcol('PHASE_DIR')[0][0]
            ra_deg = float(np.degrees(phase_dir[0])) % 360.0
            dec_deg = float(np.degrees(phase_dir[1]))

        with table(f'{ms_path}/SPECTRAL_WINDOW', readonly=True, ack=False) as spw:
            ref_freq = float(spw.getcol('REF_FREQUENCY')[0])
            freqs = spw.getcol('CHAN_FREQ')[0]
            freq_min = float(freqs.min())
            freq_max = float(freqs.max())

        return cls(
            ms_path=ms_path,
            time_range_sec=float(time_range),
            n_timesteps=n_timesteps,
            integration_time_sec=integration_time,
            ra_center_deg=ra_deg,
            dec_center_deg=dec_deg,
            reference_freq_hz=ref_freq,
            freq_min_hz=freq_min,
            freq_max_hz=freq_max
        )


@dataclass
class TransientConfig:
    """Configuration for generating transient sources."""
    fov_deg: float = 2.0
    duration_max_sec: float = 180.0
    shapes: List[str] = field(default_factory=lambda: ['gaussian', 'exponential', 'step'])
    flux_range_jy: Tuple[float, float] = (0.1, 5.0)
    spectral_index_range: Tuple[float, float] = (-2.0, 0.0)
    periodic_fraction: float = 0.0
    period_range_sec: Optional[Tuple[float, float]] = None


@dataclass
class HCIConfig:
    """Configuration for pfb-imaging HCI injection."""
    nx: int = 3072
    ny: int = 3072
    cell_size: float = 2.4
    fov: float = 2.0
    product: str = 'I'
    nworkers: int = 16
    robustness: float = 0.0
    data_column: str = 'CORRECTED_DATA-MODEL_DATA'
    weight_column: str = 'WEIGHT_SPECTRUM'
    output_format: str = 'zarr'


@dataclass
class Transient:
    """A single transient source."""
    name: str
    ra_deg: float
    dec_deg: float
    peak_time_sec: float
    duration_sec: float
    shape: str
    peak_flux_jy: float
    spectral_index: float
    reference_freq_hz: float
    periodic: bool = False
    period_sec: Optional[float] = None
    expected_cube_flux_jy: Optional[float] = None

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to YAML-compatible dictionary format."""
        return {
            'name': self.name,
            'position': {'ra': self.ra_deg, 'dec': self.dec_deg},
            'time': {'peak_time': self.peak_time_sec, 'duration': self.duration_sec, 'shape': self.shape},
            'frequency': {'peak_flux': self.peak_flux_jy, 'reference_freq': self.reference_freq_hz, 'spectral_index': self.spectral_index},
            'periodicity': {'enabled': self.periodic, 'period': self.period_sec} if self.periodic else {'enabled': False}
        }


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
        duration_min = obs.integration_time_sec
        duration_max = min(cfg.duration_max_sec, obs.time_range_sec)
        transients = []

        for i in range(nsources):
            ra = (obs.ra_center_deg + rng.uniform(-half_fov, half_fov)) % 360.0
            dec = np.clip(obs.dec_center_deg + rng.uniform(-half_fov, half_fov), -90.0, 90.0)
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
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            yaml.dump({'transients': [t.to_yaml_dict() for t in self.transients]}, f, sort_keys=False)
        return output_path

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
        cfg = hci_config or HCIConfig()
        output_dir = Path(output_dir)
        transient_yaml = Path(transient_yaml)

        cmd = [
            'pfb', 'hci',
            '--ms', self.obs.ms_path,
            '-o', str(output_dir),
            '--output-format', cfg.output_format,
            '--data-column', cfg.data_column,
            '--weight-column', cfg.weight_column,
            '--nx', str(cfg.nx),
            '--ny', str(cfg.ny),
            '--cell-size', str(cfg.cell_size),
            '-fov', str(cfg.fov),
            '--product', cfg.product,
            '--nworkers', str(cfg.nworkers),
            '--robustness', str(cfg.robustness),
            '--overwrite',
            '--stack',
            '--inject-transients', str(transient_yaml)
        ]

        # Build shell command with optional venv activation
        cmd_str = ' '.join(cmd)
        if venv_path:
            cmd_str = f". {venv_path}/bin/activate && {cmd_str}"

        start_time = datetime.now()
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, executable='/bin/bash')
        elapsed = (datetime.now() - start_time).total_seconds()

        run_info = {
            'output_dir': str(output_dir),
            'transient_yaml': str(transient_yaml),
            'elapsed_sec': elapsed,
            'returncode': result.returncode,
            'success': result.returncode == 0
        }

        if result.returncode != 0:
            run_info['stderr'] = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr

        # Update metadata
        self.metadata['hci_run'] = run_info
        return run_info


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
    # HCI options
    parser.add_argument('--run-hci', action='store_true', help='Run HCI injection after generating transients')
    parser.add_argument('--hci-output', help='Output directory for HCI zarr cube')
    parser.add_argument('--venv', help='Path to virtualenv (e.g. ~/venvs/sim_env)')
    parser.add_argument('--nx', type=int, default=3072, help='Image size x')
    parser.add_argument('--ny', type=int, default=3072, help='Image size y')
    parser.add_argument('--cell-size', type=float, default=2.4, help='Cell size (arcsec)')
    parser.add_argument('--nworkers', type=int, default=16, help='Number of workers')
    args = parser.parse_args()

    # Check FOV against cube size
    max_fov_deg = min(args.nx, args.ny) * args.cell_size / 3600.0
    if args.fov > max_fov_deg:
        print(f"Warning: Requested FOV ({args.fov}°) exceeds cube size ({max_fov_deg:.2f}°)")
        print(f"  Clamping FOV to {max_fov_deg:.2f}° (cube: {args.nx}x{args.ny} @ {args.cell_size}\")")
        args.fov = max_fov_deg

    cfg = TransientConfig(
        fov_deg=args.fov,
        flux_range_jy=(args.flux_min, args.flux_max),
        duration_max_sec=args.duration_max
    )
    sim = TransientSimulator.from_ms(args.ms, cfg)

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
        sim.save(args.output)
        print(sim.summary())
        transient_yaml_path = Path(args.output)

    if args.run_hci:
        if not args.hci_output:
            args.hci_output = str(transient_yaml_path.with_suffix('.zarr'))
        hci_cfg = HCIConfig(nx=args.nx, ny=args.ny, cell_size=args.cell_size,
                            fov=args.fov, nworkers=args.nworkers)
        print(f"Running HCI injection -> {args.hci_output}")
        result = sim.run_hci(args.hci_output, str(transient_yaml_path), hci_cfg, args.venv)
        if result['success']:
            print(f"HCI complete in {result['elapsed_sec']:.1f}s")
        else:
            print(f"HCI failed: {result.get('stderr', 'unknown error')}")

    if args.manifest and sim.transients:
        sim.save_manifest(args.manifest)
        print(f"Manifest saved to {args.manifest}")
        # Auto-generate region file
        reg_path = str(Path(args.manifest).with_suffix('.reg'))
        count = manifest_to_regions(args.manifest, reg_path)
        print(f"Region file saved to {reg_path} ({count} regions)")
