"""One fingerprinted parameter box shared by every experimental stage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ParameterBox:
    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.names or not (len(self.names) == len(self.lower) == len(self.upper)):
            raise ValueError("Parameter names and bounds must have the same positive length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Parameter names must be unique")
        if not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper)):
            raise ValueError("Parameter bounds must be finite")
        if np.any(np.asarray(self.lower) >= np.asarray(self.upper)):
            raise ValueError("Every lower parameter bound must be below its upper bound")

    @classmethod
    def from_sequences(
        cls,
        names: Sequence[str],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> "ParameterBox":
        return cls(tuple(names), tuple(float(x) for x in lower), tuple(float(x) for x in upper))

    @property
    def dimension(self) -> int:
        return len(self.names)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            {"names": self.names, "lower": self.lower, "upper": self.upper},
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(canonical).hexdigest()

    def contains(self, physical: np.ndarray | Sequence[float]) -> np.ndarray:
        values = np.asarray(physical, dtype=float)
        self._require_last_dimension(values)
        lower = np.asarray(self.lower)
        upper = np.asarray(self.upper)
        return np.all((values >= lower) & (values <= upper), axis=-1)

    def require_physical(self, physical: np.ndarray | Sequence[float]) -> np.ndarray:
        values = np.asarray(physical, dtype=float)
        self._require_last_dimension(values)
        if not np.all(np.isfinite(values)):
            raise ValueError("Physical parameter values must be finite")
        valid = self.contains(values)
        if not np.all(valid):
            bad = values if values.ndim == 1 else values[np.flatnonzero(~valid)[0]]
            violations = [
                f"{name}={value:.17g} not in [{lo:.17g}, {hi:.17g}]"
                for name, value, lo, hi in zip(
                    self.names, bad, self.lower, self.upper, strict=True
                )
                if value < lo or value > hi
            ]
            raise ValueError("Input is outside the authoritative parameter box: " + "; ".join(violations))
        return values

    def to_canonical(self, physical: np.ndarray | Sequence[float]) -> np.ndarray:
        values = self.require_physical(physical)
        lower = np.asarray(self.lower)
        upper = np.asarray(self.upper)
        return 2.0 * (values - lower) / (upper - lower) - 1.0

    def to_physical(self, canonical: np.ndarray | Sequence[float]) -> np.ndarray:
        values = np.asarray(canonical, dtype=float)
        self._require_last_dimension(values)
        if not np.all(np.isfinite(values)):
            raise ValueError("Canonical parameter values must be finite")
        tolerance = 32.0 * np.finfo(float).eps
        if np.any(values < -1.0 - tolerance) or np.any(values > 1.0 + tolerance):
            raise ValueError("Canonical input lies outside [-1, 1]")
        values = np.clip(values, -1.0, 1.0)
        lower = np.asarray(self.lower)
        upper = np.asarray(self.upper)
        return 0.5 * (values + 1.0) * (upper - lower) + lower

    def _require_last_dimension(self, values: np.ndarray) -> None:
        if values.ndim < 1 or values.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected final parameter dimension {self.dimension}; got shape {values.shape}"
            )

