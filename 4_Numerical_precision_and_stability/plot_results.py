import csv
import matplotlib.pyplot as plt
import numpy as np


def load_data(filepath="benchmark_results.csv"):
    rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "operation": row["operation"],
                    "n": int(row["n"]),
                    "dtype": row["dtype"],
                    "median_time_s": float(row["median_time_s"]),
                    "min_time_s": float(row["min_time_s"]),
                    "gflops": float(row["gflops"]),
                    "bandwidth_gb_s": float(row["bandwidth_gb_s"]),
                    "intensity": float(row["intensity"]),
                }
            )
    return rows


def plot_roofline(
    data,
    p_peak_fp32=364.0,
    p_peak_fp64=171.4,
    b_mem=42.5,
    filename="roofline_model.png",
):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Range of Arithmetic Intensity
    i_vals = np.logspace(-2, 3, 500)

    # Theoretical ceiling curves
    p_ceil_fp64 = np.minimum(p_peak_fp64, b_mem * i_vals)
    p_ceil_fp32 = np.minimum(p_peak_fp32, b_mem * i_vals)

    ax.plot(
        i_vals,
        p_ceil_fp64,
        "k-",
        linewidth=2,
        label=f"fp64 Ceiling (Ppeak={p_peak_fp64:.0f} GFLOP/s, B={b_mem:.0f} GB/s)",
    )
    ax.plot(
        i_vals,
        p_ceil_fp32,
        "k--",
        linewidth=1.5,
        label=f"fp32 Ceiling (Ppeak={p_peak_fp32:.0f} GFLOP/s)",
    )

    ridge_fp64 = p_peak_fp64 / b_mem
    ax.axvline(
        ridge_fp64,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"fp64 Ridge Point ({ridge_fp64:.1f} FLOP/B)",
    )

    markers = {
        ("gemv", "float64"): ("o", "tab:blue", "GEMV (float64)"),
        ("gemv", "float32"): ("s", "tab:cyan", "GEMV (float32)"),
        ("gemm", "float64"): ("^", "tab:red", "GEMM (float64)"),
        ("gemm", "float32"): ("D", "tab:orange", "GEMM (float32)"),
    }

    for (op, dt), (m, c, lbl) in markers.items():
        subset = [d for d in data if d["operation"] == op and d["dtype"] == dt]
        if not subset:
            continue
        intensities = [d["intensity"] for d in subset]
        gflops = [d["gflops"] for d in subset]
        ax.scatter(
            intensities,
            gflops,
            marker=m,
            color=c,
            edgecolors="black",
            s=65,
            label=lbl,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        "Arithmetic Intensity $I$ [FLOP/byte]", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel(
        "Computational Performance $P$ [GFLOP/s]",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        "Empirical Roofline Model Analysis (GEMV vs. GEMM)",
        fontsize=13,
        pad=12,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_scaling_metrics(data, filename="performance_scaling.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # 1. Execution time vs N (Log-Log)
    ax = axes[0]
    for op, ls in [("gemv", "--"), ("gemm", "-")]:
        for dt, color in [("float64", "tab:blue"), ("float32", "tab:orange")]:
            subset = [
                d for d in data if d["operation"] == op and d["dtype"] == dt
            ]
            ax.loglog(
                [d["n"] for d in subset],
                [d["median_time_s"] for d in subset],
                marker="o",
                linestyle=ls,
                color=color,
                label=f"{op.upper()} ({dt})",
            )
    ax.set_xlabel("Problem Size $N$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time [s]", fontsize=11, fontweight="bold")
    ax.set_title("Time Scaling vs. $N$", fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()

    # 2. Performance (GFLOP/s) vs N
    ax = axes[1]
    for op, ls in [("gemv", "--"), ("gemm", "-")]:
        for dt, color in [("float64", "tab:blue"), ("float32", "tab:orange")]:
            subset = [
                d for d in data if d["operation"] == op and d["dtype"] == dt
            ]
            ax.plot(
                [d["n"] for d in subset],
                [d["gflops"] for d in subset],
                marker="o",
                linestyle=ls,
                color=color,
                label=f"{op.upper()} ({dt})",
            )
    ax.set_xlabel("Problem Size $N$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Performance [GFLOP/s]", fontsize=11, fontweight="bold")
    ax.set_title("Throughput vs. $N$", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 3. Effective Bandwidth (GB/s) vs N
    ax = axes[2]
    for op, ls in [("gemv", "--"), ("gemm", "-")]:
        for dt, color in [("float64", "tab:blue"), ("float32", "tab:orange")]:
            subset = [
                d for d in data if d["operation"] == op and d["dtype"] == dt
            ]
            ax.loglog(
                [d["n"] for d in subset],
                [d["bandwidth_gb_s"] for d in subset],
                marker="s",
                linestyle=ls,
                color=color,
                label=f"{op.upper()} ({dt})",
            )
    ax.set_xlabel("Problem Size $N$", fontsize=11, fontweight="bold")
    ax.set_ylabel(
        "Effective Bandwidth [GB/s]", fontsize=11, fontweight="bold"
    )
    ax.set_title("Effective Bandwidth vs. $N$", fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    benchmark_data = load_data()
    plot_roofline(benchmark_data)
    plot_scaling_metrics(benchmark_data)
    print("Plots generated: roofline_model.png and performance_scaling.png")