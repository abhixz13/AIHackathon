from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""

    return f"{value * 100:.{decimals}f}%"


def pretty_print_metrics(metrics: Mapping[str, float]) -> str:
    """Return a pretty string representation of metric values."""

    if not metrics:
        return "<no metrics>"
    longest = max(len(k) for k in metrics)
    lines = []
    for name, value in metrics.items():
        lines.append(f"{name.ljust(longest)} : {value:.4f}")
    result = "\n".join(lines)
    print(result)
    return result


def save_config(config: Dict, path: str) -> None:
    """Save a JSON configuration file."""

    Path(path).write_text(json.dumps(config, indent=2))


def load_config(path: str) -> Dict:
    """Load a JSON configuration file."""

    return json.loads(Path(path).read_text())


def validate_schema(data: Mapping, required_fields: Iterable[str]) -> None:
    """Validate that *data* contains all *required_fields*.

    Raises
    ------
    KeyError
        If a required field is missing.
    """

    missing = [f for f in required_fields if f not in data]
    if missing:
        raise KeyError(f"Missing required fields: {', '.join(missing)}")
