from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

from cloud_column_reproduction.mcmc.gpu_batching import PairedMicrobatchPredictor


def test_paired_calls_are_concatenated_once_and_split_exactly() -> None:
    shapes: list[tuple[int, ...]] = []

    def predictor(values: np.ndarray) -> np.ndarray:
        shapes.append(values.shape)
        return np.column_stack((values[:, 0], values[:, 0] ** 2))

    broker = PairedMicrobatchPredictor(predictor, window_seconds=0.1)
    first = np.arange(16, dtype=float)[:, None]
    second = np.arange(16, 32, dtype=float)[:, None]
    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(broker, first)
        time.sleep(0.005)
        right = pool.submit(broker, second)
        left_value = left.result()
        right_value = right.result()
    np.testing.assert_array_equal(left_value[:, 0], first[:, 0])
    np.testing.assert_array_equal(right_value[:, 0], second[:, 0])
    assert shapes == [(32, 1)]
    assert broker.metadata["microbatch_paired_calls"] == "2"
    broker.close()


def test_unmatched_call_uses_bounded_singleton_fallback() -> None:
    broker = PairedMicrobatchPredictor(
        lambda values: values + 1.0, window_seconds=0.001
    )
    values = np.arange(6, dtype=float).reshape(3, 2)
    np.testing.assert_array_equal(broker(values), values + 1.0)
    assert broker.metadata["microbatch_singleton_calls"] == "1"
    broker.close()


def test_initialization_and_inference_share_the_same_owning_thread():
    import threading

    owner = {}
    def predict(values):
        assert threading.get_ident() == owner["thread"]
        return values
    broker = PairedMicrobatchPredictor(predict, window_seconds=0.001)
    broker.call_on_worker(lambda: owner.update(thread=threading.get_ident()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(broker, np.ones((16, 2))) for _ in range(2)]
        for future in futures:
            np.testing.assert_array_equal(future.result(), np.ones((16, 2)))
    broker.close()


def test_runtime_closes_gpu_resources_before_owning_worker_exits():
    import threading
    from cloud_column_reproduction.mcmc.runtime import RuntimeEvaluator

    closed_on = []
    class Resource:
        def close(self):
            closed_on.append(threading.get_ident())
    broker = PairedMicrobatchPredictor(lambda values: values)
    worker = broker.call_on_worker(threading.get_ident)
    runtime = RuntimeEvaluator(None, None, True, None, {}, resources=(broker, Resource()))
    runtime.close()
    runtime.close()
    assert closed_on == [worker]
