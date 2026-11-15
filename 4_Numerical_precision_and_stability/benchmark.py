import time
import json
import numpy as np


def benchmark_operation(op_type, n, dtype, warmups=3, repetitions=10):
    """
    Benchmarks NumPy BLAS kernels excluding memory allocation overhead.
    """
    bytes_per_elem = np.dtype(dtype).itemsize

    # Allocation and initialization outside timed region
    if op_type == "gemv":
        a = np.full((n, n), 1.0, dtype=dtype)
        x = np.full(n, 3.0, dtype=dtype)
        y = np.empty(n, dtype=dtype)

        flops = 2.0 * (n**2)
        bytes_transferred = (n**2 + 2 * n) * bytes_per_elem

        # Warm-up phase (stabilizes frequency, cache, and thread pools)
        for _ in range(warmups):
            np.dot(a, x, out=y)

        # Timed benchmark iterations
        times = []
        for _ in range(repetitions):
            t_start = time.perf_counter()
            np.dot(a, x, out=y)
            t_end = time.perf_counter()
            times.append(t_end - t_start)

    elif op_type == "gemm":
        a = np.full((n, n), 1.0, dtype=dtype)
        b = np.full((n, n), 2.0, dtype=dtype)
        c = np.empty((n, n), dtype=dtype)

        flops = 2.0 * (n**3)
        bytes_transferred = 3.0 * (n**2) * bytes_per_elem

        # Warm-up phase
        for _ in range(warmups):
            np.matmul(a, b, out=c)

        # Timed benchmark iterations
        times = []
        for _ in range(repetitions):
            t_start = time.perf_counter()
            np.matmul(a, b, out=c)
            t_end = time.perf_counter()
            times.append(t_end - t_start)
    else:
        raise ValueError(f"Unknown operation type: {op_type}")

    # Use median to filter operating system interrupts and context switches
    median_time = float(np.median(times))
    min_time = float(np.min(times))

    gflops = (flops / median_time) * 1e-9
    bandwidth_gb_s = (bytes_transferred / median_time) * 1e-9
    intensity = flops / bytes_transferred

    return {
        "operation": op_type,
        "n": n,
        "dtype": np.dtype(dtype).name,
        "median_time_s": median_time,
        "min_time_s": min_time,
        "gflops": gflops,
        "bandwidth_gb_s": bandwidth_gb_s,
        "intensity": intensity,
    }


def run_benchmark_suite():
    # Power-of-two and intermediate sizes to observe cache boundary crossings
    sizes_gemv = [256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
    sizes_gemm = [128, 256, 512, 1024, 1536, 2048, 3072, 4096]

    precisions = [np.float32, np.float64]
    results = []

    print("=" * 70)
    print("STARTING ROOFLINE BENCHMARK SUITE")
    print("=" * 70)

    # 1. Benchmark Matrix-Vector (GEMV)
    for dt in precisions:
        dt_name = np.dtype(dt).name
        print(f"\n--- Running GEMV ({dt_name}) ---")
        for n in sizes_gemv:
            # Adjust repetition count for very large sizes to limit runtime
            reps = 15 if n <= 4096 else 5
            res = benchmark_operation("gemv", n, dt, repetitions=reps)
            results.append(res)
            print(
                f"N = {n:5d} | Time: {res['median_time_s']*1e3:8.2f} ms | "
                f"Perf: {res['gflops']:7.2f} GFLOP/s | "
                f"BW: {res['bandwidth_gb_s']:7.2f} GB/s | "
                f"Intensity: {res['intensity']:.3f} FLOP/B"
            )

    # 2. Benchmark Matrix-Matrix (GEMM)
    for dt in precisions:
        dt_name = np.dtype(dt).name
        print(f"\n--- Running GEMM ({dt_name}) ---")
        for n in sizes_gemm:
            reps = 10 if n <= 1024 else 3
            res = benchmark_operation("gemm", n, dt, repetitions=reps)
            results.append(res)
            print(
                f"N = {n:5d} | Time: {res['median_time_s']*1e3:8.2f} ms | "
                f"Perf: {res['gflops']:7.2f} GFLOP/s | "
                f"BW: {res['bandwidth_gb_s']:7.2f} GB/s | "
                f"Intensity: {res['intensity']:7.2f} FLOP/B"
            )

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nBenchmark completed. Results exported to benchmark_results.json")


if __name__ == "__main__":
    run_benchmark_suite()