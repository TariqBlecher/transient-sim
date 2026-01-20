"""Transient dataclasses and YAML I/O.

Central domain model for transient sources, with factory methods
for creating transients from YAML, manifest rows, or programmatically.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union


@dataclass
class TransientConfig:
    """Configuration for generating transient sources."""

    fov_deg: float = 2.0
    duration_max_sec: float = 180.0
    shapes: List[str] = field(
        default_factory=lambda: ["gaussian", "exponential", "step"]
    )
    flux_range_jy: tuple = (0.1, 5.0)
    spectral_index_range: tuple = (-2.0, 0.0)
    periodic_fraction: float = 0.0
    period_range_sec: Optional[tuple] = None


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

    def __post_init__(self):
        """Validate transient configuration."""
        if self.periodic and self.period_sec is None:
            raise ValueError(
                f"Transient '{self.name}': period_sec must be set when periodic=True"
            )

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to YAML-compatible dictionary format for pfb-imaging."""
        result = {
            "name": self.name,
            "position": {"ra": self.ra_deg, "dec": self.dec_deg},
            "time": {
                "peak_time": self.peak_time_sec,
                "duration": self.duration_sec,
                "shape": self.shape,
            },
            "frequency": {
                "peak_flux": self.peak_flux_jy,
                "reference_freq": self.reference_freq_hz,
                "spectral_index": self.spectral_index,
            },
        }
        if self.periodic:
            result["periodicity"] = {"enabled": True, "period": self.period_sec}
        else:
            result["periodicity"] = {"enabled": False}
        return result

    @classmethod
    def from_yaml_dict(cls, d: Dict[str, Any]) -> "Transient":
        """Create Transient from YAML dictionary format.

        Args:
            d: Dictionary with 'position', 'time', 'frequency' sub-dicts

        Returns:
            Transient instance
        """
        periodicity = d.get("periodicity", {})
        return cls(
            name=d["name"],
            ra_deg=d["position"]["ra"],
            dec_deg=d["position"]["dec"],
            peak_time_sec=d["time"]["peak_time"],
            duration_sec=d["time"]["duration"],
            shape=d["time"]["shape"],
            peak_flux_jy=d["frequency"]["peak_flux"],
            spectral_index=d["frequency"].get("spectral_index", 0.0),
            reference_freq_hz=d["frequency"]["reference_freq"],
            periodic=periodicity.get("enabled", False),
            period_sec=periodicity.get("period"),
        )

    @classmethod
    def from_manifest_row(cls, row: Dict[str, Any]) -> "Transient":
        """Create Transient from a manifest DataFrame row (as dict).

        Args:
            row: Dictionary with flat column names matching dataclass fields

        Returns:
            Transient instance
        """
        return cls(
            name=row["name"],
            ra_deg=row["ra_deg"],
            dec_deg=row["dec_deg"],
            peak_time_sec=row["peak_time_sec"],
            duration_sec=row["duration_sec"],
            shape=row["shape"],
            peak_flux_jy=float(row["peak_flux_jy"]),
            spectral_index=row.get("spectral_index", 0.0),
            reference_freq_hz=row["reference_freq_hz"],
            periodic=row.get("periodic", False),
            period_sec=row.get("period_sec"),
            expected_cube_flux_jy=row.get("expected_cube_flux_jy"),
        )


def save_transients_yaml(
    transients: List[Transient], output_path: Union[str, Path]
) -> Path:
    """Save transients to YAML file for pfb-imaging.

    Args:
        transients: List of Transient objects
        output_path: Output file path

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    data = {"transients": [t.to_yaml_dict() for t in transients]}
    with open(output_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    return output_path


def load_transients_yaml(yaml_path: Union[str, Path]) -> List[Transient]:
    """Load transients from a YAML file.

    Args:
        yaml_path: Path to YAML file with 'transients' list

    Returns:
        List of Transient objects
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return [Transient.from_yaml_dict(t) for t in data["transients"]]
