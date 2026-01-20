"""HCI (pfb-imaging) subprocess orchestration.

Provides configuration and execution wrapper for pfb hci command.
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, Union


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


def run_hci(
    ms_path: Union[str, Path],
    output_dir: Union[str, Path],
    transient_yaml: Optional[Union[str, Path]] = None,
    hci_config: Optional[HCIConfig] = None,
    venv_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run pfb HCI imaging, optionally injecting transients.

    Args:
        ms_path: Path to Measurement Set
        output_dir: Output directory for zarr cube
        transient_yaml: Path to transient YAML file (None for baseline cube without transients)
        hci_config: HCI configuration (uses defaults if None)
        venv_path: Path to virtualenv to activate (e.g. ~/venvs/sim_env)

    Returns:
        Dict with:
        - output_dir: Output path
        - transient_yaml: YAML path used (or None if baseline)
        - elapsed_sec: Runtime in seconds
        - returncode: Process return code
        - success: Boolean success flag
        - stderr: Truncated stderr if failed
    """
    cfg = hci_config or HCIConfig()
    ms_path = str(ms_path)
    output_dir = Path(output_dir)

    cmd = [
        'pfb', 'hci',
        '--ms', ms_path,
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
    ]

    # Only add transient injection if YAML provided
    if transient_yaml is not None:
        transient_yaml = Path(transient_yaml)
        cmd.extend(['--inject-transients', str(transient_yaml)])

    # Build shell command with optional venv activation
    cmd_str = ' '.join(cmd)
    if venv_path:
        cmd_str = f". {venv_path}/bin/activate && {cmd_str}"

    start_time = datetime.now()
    result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, executable='/bin/bash')
    elapsed = (datetime.now() - start_time).total_seconds()

    run_info = {
        'output_dir': str(output_dir),
        'transient_yaml': str(transient_yaml) if transient_yaml else None,
        'elapsed_sec': elapsed,
        'returncode': result.returncode,
        'success': result.returncode == 0
    }

    if result.returncode != 0:
        run_info['stderr'] = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr

    return run_info
