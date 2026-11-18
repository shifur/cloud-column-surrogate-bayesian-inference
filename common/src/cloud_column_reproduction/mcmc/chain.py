"""Atomic, unthinned per-ensemble chain serialization."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from cloud_column_reproduction.provenance import sha256_file

from .sampler import EnsembleResult


def write_chain_netcdf(
    path: str | Path,
    result: EnsembleResult,
    *,
    parameter_names: tuple[str, ...],
    metadata: dict[str, object],
    retained_predictions: np.ndarray | None = None,
    prediction_cache_binding: dict[str, str] | None = None,
) -> dict[str, object]:
    try:
        import xarray as xr
    except ImportError as error:
        raise RuntimeError("xarray and h5netcdf are required for chain serialization") from error
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    transitions, walkers, parameters = result.positions.shape
    data_vars = {
            "position": (("transition", "walker", "parameter"), result.positions),
            "log_prior": (("transition", "walker"), result.log_prior),
            "log_likelihood": (("transition", "walker"), result.log_likelihood),
            "log_posterior": (("transition", "walker"), result.log_posterior),
            "evaluation_status": (("transition", "walker"), result.status),
            "accepted": (("transition", "walker"), result.accepted),
            "initial_position": (("walker", "parameter"), result.initial_positions),
            "post_burn_position": (("walker", "parameter"), result.post_burn_positions),
        }
    cache_sha256 = None
    if retained_predictions is not None:
        cache = np.asarray(retained_predictions)
        if cache.shape != (transitions, walkers, 6) or cache.dtype != np.float64 or not np.all(np.isfinite(cache)):
            raise ValueError("SCCM retained predictions must be finite float64 (transition, walker, 6)")
        required = {"chain_config_sha256", "evaluator_sha256", "observation_sha256", "parameter_box_sha256",
                    "sccm_executable_sha256", "sccm_namelist_sha256", "sccm_rendered_namelist_sha256",
                    "sccm_radiation_table_sha256", "prediction_source"}
        if prediction_cache_binding is None or required.difference(prediction_cache_binding):
            raise ValueError("SCCM prediction cache is missing required provenance bindings")
        cache_sha256 = __import__("hashlib").sha256(cache.tobytes(order="C")).hexdigest()
        data_vars["sccm_prediction_cache"] = (("transition", "walker", "output"), cache)
        metadata = {**metadata, "prediction_cache_policy": "all-retained-production-states",
                    "prediction_cache_dtype": "float64", "prediction_cache_array_sha256": cache_sha256,
                    "prediction_cache_binding": prediction_cache_binding}
    elif prediction_cache_binding is not None:
        raise ValueError("Prediction-cache binding supplied without predictions")
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "transition": np.arange(transitions),
            "walker": np.arange(walkers),
            "parameter": np.asarray(parameter_names),
            **({"output": np.asarray(("PCP", "ACC", "LWP", "IWP", "OLR", "OSR"))} if retained_predictions is not None else {}),
        },
        attrs={key: _attribute(value) for key, value in metadata.items()},
    )
    encoding = {
        "position": {"zlib": True, "complevel": 2, "chunksizes": (min(2048, transitions), walkers, parameters)},
        "log_prior": {"zlib": True, "complevel": 2, "chunksizes": (min(4096, transitions), walkers)},
        "log_likelihood": {"zlib": True, "complevel": 2, "chunksizes": (min(4096, transitions), walkers)},
        "log_posterior": {"zlib": True, "complevel": 2, "chunksizes": (min(4096, transitions), walkers)},
        **({"sccm_prediction_cache": {"zlib": True, "complevel": 2, "chunksizes": (min(2048, transitions), walkers, 6)}} if retained_predictions is not None else {}),
    }
    try:
        dataset.to_netcdf(temporary, engine="h5netcdf", format="NETCDF4", encoding=encoding)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": [transitions, walkers, parameters],
        "thinning": "none",
        "prediction_cache_present": retained_predictions is not None,
        "prediction_cache_array_sha256": cache_sha256,
    }


def read_chain_netcdf(path: str | Path) -> dict[str, np.ndarray]:
    try:
        import xarray as xr
    except ImportError as error:
        raise RuntimeError("xarray and h5netcdf are required for chain serialization") from error
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        return {name: np.asarray(dataset[name].values) for name in dataset.data_vars}


def write_chain_netcdf_from_blocks(
    path: str | Path,
    block_paths: tuple[Path, ...] | list[Path],
    *,
    parameter_names: tuple[str, ...],
    initial_positions: np.ndarray,
    post_burn_positions: np.ndarray,
    metadata: dict[str, object],
    include_sccm_predictions: bool = False,
    sampler_coordinate_system: str = "physical",
    parameter_lower: np.ndarray | None = None,
    parameter_upper: np.ndarray | None = None,
) -> dict[str, object]:
    """Consolidate immutable production blocks without loading the full chain.

    The temporary NetCDF is populated one committed block at a time and is
    atomically renamed only after every source block has been copied.
    """
    try:
        import h5netcdf
        import h5py
    except ImportError as error:
        raise RuntimeError("h5netcdf and h5py are required for chain finalization") from error
    paths = tuple(Path(item).resolve() for item in block_paths)
    if not paths:
        raise ValueError("At least one committed production block is required")
    if sampler_coordinate_system not in {"physical", "box-logit"}:
        raise ValueError("Unknown sampler coordinate system")
    if sampler_coordinate_system == "box-logit":
        if parameter_lower is None or parameter_upper is None:
            raise ValueError("Box-logit chain finalization requires parameter bounds")
        lower = np.asarray(parameter_lower, dtype=np.float64)
        upper = np.asarray(parameter_upper, dtype=np.float64)
        from .transforms import logit_to_physical
    else:
        lower = upper = None
    counts: list[int] = []
    for source in paths:
        with np.load(source, allow_pickle=False) as archive:
            position = np.asarray(archive["position"])
            if position.ndim != 3 or position.shape[1:] != (32, len(parameter_names)):
                raise ValueError(f"Invalid committed position block: {source}")
            counts.append(len(position))
            if include_sccm_predictions and "sccm_prediction_cache" not in archive:
                raise ValueError("SCCM finalization requires aligned prediction blocks")
    total = int(sum(counts)); walkers = 32; parameters = len(parameter_names)
    destination = Path(path).resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        with h5netcdf.File(temporary, "w") as dataset:
            dataset.dimensions = {
                "transition": total, "walker": walkers, "parameter": parameters,
                **({"output": 6} if include_sccm_predictions else {}),
            }
            dataset.attrs.update({key: _attribute(value) for key, value in metadata.items()})
            transition = dataset.create_variable("transition", ("transition",), dtype="i8")
            transition[:] = np.arange(total, dtype=np.int64)
            walker = dataset.create_variable("walker", ("walker",), dtype="i4")
            walker[:] = np.arange(walkers, dtype=np.int32)
            parameter = dataset.create_variable("parameter", ("parameter",), dtype=string_dtype)
            parameter[:] = np.asarray(parameter_names, dtype=object)
            if include_sccm_predictions:
                output = dataset.create_variable("output", ("output",), dtype=string_dtype)
                output[:] = np.asarray(("PCP", "ACC", "LWP", "IWP", "OLR", "OSR"), dtype=object)
            variables = {
                "position": dataset.create_variable("position", ("transition", "walker", "parameter"), dtype="f8", chunks=(min(2048,total),walkers,parameters), compression="gzip", compression_opts=2),
                "log_prior": dataset.create_variable("log_prior", ("transition", "walker"), dtype="f8", chunks=(min(4096,total),walkers), compression="gzip", compression_opts=2),
                "log_likelihood": dataset.create_variable("log_likelihood", ("transition", "walker"), dtype="f8", chunks=(min(4096,total),walkers), compression="gzip", compression_opts=2),
                "log_posterior": dataset.create_variable("log_posterior", ("transition", "walker"), dtype="f8", chunks=(min(4096,total),walkers), compression="gzip", compression_opts=2),
                "evaluation_status": dataset.create_variable("evaluation_status", ("transition", "walker"), dtype="i1", chunks=(min(4096,total),walkers), compression="gzip", compression_opts=2),
                "accepted": dataset.create_variable("accepted", ("transition", "walker"), dtype="i1", chunks=(min(4096,total),walkers), compression="gzip", compression_opts=2),
            }
            initial = dataset.create_variable("initial_position", ("walker", "parameter"), dtype="f8")
            initial[:] = np.asarray(initial_positions, dtype=np.float64)
            post_burn = dataset.create_variable("post_burn_position", ("walker", "parameter"), dtype="f8")
            post_burn[:] = np.asarray(post_burn_positions, dtype=np.float64)
            if include_sccm_predictions:
                variables["sccm_prediction_cache"] = dataset.create_variable(
                    "sccm_prediction_cache", ("transition", "walker", "output"), dtype="f8",
                    chunks=(min(2048,total),walkers,6), compression="gzip", compression_opts=2,
                )
            prediction_digest = hashlib.sha256() if include_sccm_predictions else None
            offset = 0
            for source, count in zip(paths, counts, strict=True):
                with np.load(source, allow_pickle=False) as archive:
                    for name, variable in variables.items():
                        values = np.asarray(archive[name])
                        if sampler_coordinate_system == "box-logit":
                            if name == "position":
                                values = logit_to_physical(values, lower, upper)
                            elif name == "log_prior":
                                values = np.where(
                                    np.asarray(archive["evaluation_status"]) == 0,
                                    0.0,
                                    -np.inf,
                                )
                            elif name == "log_posterior":
                                values = np.asarray(archive["log_likelihood"])
                        variable[offset:offset+count] = values
                        if name == "sccm_prediction_cache" and prediction_digest is not None:
                            prediction_digest.update(np.ascontiguousarray(values).tobytes(order="C"))
                offset += count
        os.replace(temporary, destination)
    finally:
        if temporary.exists(): temporary.unlink()
    return {
        "path": str(destination), "sha256": sha256_file(destination),
        "shape": [total, walkers, parameters], "thinning": "none",
        "source_block_count": len(paths),
        "prediction_cache_present": include_sccm_predictions,
        "prediction_cache_array_sha256": (
            prediction_digest.hexdigest() if prediction_digest is not None else None
        ),
    }


def _attribute(value: object) -> str | int | float:
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
