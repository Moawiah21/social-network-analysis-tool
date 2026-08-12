import time

def measure_runtime(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    runtime = end_time - start_time
    return result, runtime