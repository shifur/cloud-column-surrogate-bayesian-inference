"""Thread-safe, semantics-preserving microbatching for paired GPU ensembles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
from typing import Any, Callable

import numpy as np


@dataclass
class _Request:
    values: np.ndarray
    event: threading.Event = field(default_factory=threading.Event)
    result: np.ndarray | None = None
    error: BaseException | None = None


class PairedMicrobatchPredictor:
    """Rendezvous at most two concurrent prediction calls.

    The first caller waits for a short, bounded window. A concurrent second
    caller evaluates the concatenated rows once and returns the exact slices to
    both callers. If no partner arrives, the first call executes alone. Thus
    differing DE/Snooker proposal patterns cannot deadlock.
    """

    def __init__(
        self,
        predictor: Callable[[np.ndarray], np.ndarray],
        *,
        window_seconds: float = 0.002,
    ) -> None:
        if not np.isfinite(window_seconds) or window_seconds <= 0.0:
            raise ValueError("Microbatch rendezvous window must be positive")
        self.predictor = predictor
        self.window_seconds = float(window_seconds)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gpu-microbatch"
        )
        self._pending: _Request | None = None
        self._closed = False
        self._calls = 0
        self._paired_calls = 0
        self._singleton_calls = 0
        self._rows = 0

    def _evaluate(self, requests: tuple[_Request, ...]) -> None:
        sizes = tuple(len(request.values) for request in requests)
        try:
            combined = np.concatenate(
                tuple(request.values for request in requests), axis=0
            )
            prediction = np.asarray(self.predictor(combined))
            if prediction.ndim != 2 or prediction.shape[0] != sum(sizes):
                raise RuntimeError("Microbatched predictor returned the wrong shape")
            start = 0
            for request, size in zip(requests, sizes, strict=True):
                request.result = prediction[start : start + size]
                start += size
        except BaseException as error:
            for request in requests:
                request.error = error
        finally:
            with self._lock:
                self._calls += len(requests)
                self._rows += sum(sizes)
                if len(requests) == 2:
                    self._paired_calls += 2
                else:
                    self._singleton_calls += 1
            for request in requests:
                request.event.set()

    def call_on_worker(self, function: Callable[..., Any], *args: Any) -> Any:
        """Construct/prewarm CUDA state on the same thread that evaluates it."""
        if self._closed:
            raise RuntimeError("Microbatch predictor is closed")
        return self._executor.submit(function, *args).result()

    def __call__(self, values: np.ndarray) -> np.ndarray:
        array = np.ascontiguousarray(np.asarray(values))
        if array.ndim != 2 or len(array) == 0:
            raise ValueError("Microbatch requests must be nonempty matrices")
        request = _Request(array)
        with self._lock:
            if self._closed:
                raise RuntimeError("Microbatch predictor is closed")
            if self._pending is None:
                self._pending = request
                partner = None
            else:
                partner = self._pending
                self._pending = None
        if partner is not None:
            self._executor.submit(self._evaluate, (partner, request))
        elif not request.event.wait(self.window_seconds):
            with self._lock:
                if self._pending is request:
                    self._pending = None
                    run_singleton = True
                else:
                    run_singleton = False
            if run_singleton:
                self._executor.submit(self._evaluate, (request,))
        request.event.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("Microbatch request completed without a result")
        return request.result

    @property
    def metadata(self) -> dict[str, str]:
        with self._lock:
            return {
                "microbatch_policy": "paired-rendezvous-with-timed-singleton-fallback",
                "microbatch_maximum_requests": "2",
                "microbatch_window_seconds": f"{self.window_seconds:.12g}",
                "microbatch_calls": str(self._calls),
                "microbatch_paired_calls": str(self._paired_calls),
                "microbatch_singleton_calls": str(self._singleton_calls),
                "microbatch_rows": str(self._rows),
            }

    def close(self) -> None:
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("Cannot close microbatch predictor with a pending call")
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
