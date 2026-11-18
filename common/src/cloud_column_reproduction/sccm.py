"""Parallel-safe wrapper for one SCCM forward-model evaluation."""

from __future__ import annotations

import os

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from cloud_column_reproduction.domain import ParameterBox


@dataclass(frozen=True)
class SCCMRunner:
    executable: Path
    namelist: Path
    radiation_table: Path
    parameter_box: ParameterBox | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for label, path in (
            ("executable", self.executable),
            ("namelist", self.namelist),
            ("radiation table", self.radiation_table),
        ):
            resolved = Path(path).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"SCCM {label} not found: {resolved}")
            object.__setattr__(self, label.replace(" ", "_"), resolved)

    def run(self, parameters: np.ndarray | list[float]) -> np.ndarray:
        values = np.asarray(parameters, dtype=float)
        if values.shape != (11,):
            raise ValueError(f"SCCM expects 11 parameters; got shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SCCM parameters must be finite")
        if self.parameter_box is not None:
            self.parameter_box.require_physical(values)

        with tempfile.TemporaryDirectory(prefix="cloud-column-sccm-") as temporary:
            workdir = Path(temporary)
            input_path = workdir / "parameters.txt"
            output_path = workdir / "output.txt"
            runtime_namelist = workdir / "namelist.f90"
            shutil.copy2(self.radiation_table, workdir / "trp.dat")
            runtime_namelist.write_text(self.rendered_namelist(), encoding="ascii")
            input_path.write_text(
                " ".join(f"{value:.17g}" for value in values) + "\n",
                encoding="ascii",
            )
            completed = subprocess.run(
                [
                    str(self.executable),
                    input_path.name,
                    output_path.name,
                    runtime_namelist.name,
                ],
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "SCCM failed with return code "
                    f"{completed.returncode}: {completed.stderr.strip()}"
                )
            if not output_path.is_file():
                raise RuntimeError("SCCM returned successfully without creating its output file")
            output = np.asarray(np.loadtxt(output_path), dtype=float).reshape(-1)

        if output.shape != (36,):
            raise RuntimeError(f"SCCM should return 36 values; got shape {output.shape}")
        if not np.all(np.isfinite(output)):
            raise RuntimeError("SCCM returned non-finite values")
        return output

    def rendered_namelist(self) -> str:
        """Render RECORD1 from the authoritative box while preserving other settings."""
        template = self.namelist.read_text(encoding="ascii")
        if self.parameter_box is None:
            return template
        lower = ", ".join(f"{value:.17g}" for value in self.parameter_box.lower)
        upper = ", ".join(f"{value:.17g}" for value in self.parameter_box.upper)
        rendered, lower_count = re.subn(
            r"(?m)^(\s*pmin\s*=).*$",
            rf"\1 {lower}  ! generated from configs/parameter_box.toml",
            template,
        )
        rendered, upper_count = re.subn(
            r"(?m)^(\s*pmax\s*=).*$",
            rf"\1 {upper}  ! generated from configs/parameter_box.toml",
            rendered,
        )
        if lower_count != 1 or upper_count != 1:
            raise RuntimeError("Could not uniquely replace SCCM RECORD1 parameter bounds")
        return rendered

    def rendered_namelist_sha256(self) -> str:
        return hashlib.sha256(self.rendered_namelist().encode("ascii")).hexdigest()

    def diagnostics_at_120_minutes(self, parameters: np.ndarray | list[float]) -> np.ndarray:
        """Return PCP, ACC, LWP, IWP, OLR and OSR at 120 minutes."""
        return self.run(parameters)[18:24]


def runner_from_repository(
    repository: str | Path,
    *,
    prebuilt: bool = False,
    parameter_box: ParameterBox | None = None,
) -> SCCMRunner:
    repository = Path(repository).resolve()
    external = os.environ.get("SCCM_INSTALL_DIR")
    model_dir = Path(external).expanduser().resolve() if external else repository / "vendor" / "sccm"
    executable = (
        model_dir / "prebuilt" / "linux-x86_64" / "cloud_column_model.x"
        if prebuilt
        else model_dir / "cloud_column_model.x"
    )
    return SCCMRunner(
        executable=executable,
        namelist=model_dir / "namelist_3h_t30-180.f90",
        radiation_table=model_dir / "trp.dat",
        parameter_box=parameter_box,
    )
