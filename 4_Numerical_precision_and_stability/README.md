Implementation, benchmarking suite, and Roofline model analysis for dense Matrix-Vector (`GEMV`) and Matrix-Matrix (`GEMM`) multiplications using `float32` and `float64`.

## Files
- `benchmark.py`: Benchmarks execution time, throughput (GFLOP/s), and effective bandwidth (GB/s) across increasing matrix sizes $N$.
- `plot_results.py`: Generates the Roofline model and performance scaling plots from benchmark data.
- `benchmark_results.json`: Saved benchmark measurements.
- `roofline_model.png` & `performance_scaling.png`: Output performance figures.

## How to Run and Reproduce
1. Install dependencies:
   ```bash
   pip install numpy matplotlib

2. Run Benchmarks
   ```bash
   python benchmarks.py

4. Generate Plots
   ```bash
   python plot_results.py
