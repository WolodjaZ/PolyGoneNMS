import argparse
import os
import time
import timeit
from functools import partial
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import psutil
import ray
from dask.distributed import Client
from memory_profiler import memory_usage
from shapely.affinity import translate
from shapely.geometry import Polygon

from polygone_nms import nms as apply_nms


def generate_polygons(num_polygons: int, avg_overlap: float, max_points: int = 6):
    polygons = []
    base_points = np.random.rand(max_points, 2)
    base_poly = Polygon(base_points)

    for _ in range(num_polygons):
        poly = base_poly.buffer(np.random.rand())
        x_offset = np.random.rand() * avg_overlap
        y_offset = np.random.rand() * avg_overlap
        poly = translate(poly, x_offset, y_offset)

        class_label = np.random.randint(0, 2)
        confidence = np.random.rand()
        polygons.append((poly, class_label, confidence))

    return polygons


def benchmark_nms(nms_methods: List[str], save_dir: str = "./", num_workers: int = 4):
    num_polygons_list = [100, 500, 1000, 2000, 4000, 10000]  # , 40000]
    avg_overlap_list = [0.25, 0.5, 0.75]

    time_results = {}
    memory_results = {}

    print(
        f"Benchmark results will be created for the following methods: {nms_methods}, "
        f"with number of polygons: {num_polygons_list} "
        f"and average overlap: {avg_overlap_list}"
    )

    for num_polygons in num_polygons_list:
        for avg_overlap in avg_overlap_list:
            polygons = generate_polygons(num_polygons, avg_overlap)

            for method in nms_methods:
                print(
                    f"Benchmarking {method} with {num_polygons} polygons "
                    f"and {avg_overlap} avg overlap"
                )

                avg_time = []
                avg_memory = []
                for _ in range(5):
                    # Set distributed method if needed
                    if method == "Dask":
                        client = Client(n_workers=num_workers)
                        dist_method = method
                        nms_function = partial(apply_nms, client=client)
                        time.sleep(5)
                    elif method == "Ray":
                        ray.init(num_cpus=num_workers)
                        dist_method = method
                        nms_function = apply_nms
                        time.sleep(5)
                    else:
                        nms_function = apply_nms
                        dist_method = None

                    # Start measuring time and memory
                    start_time = timeit.default_timer()
                    mem_usage = memory_usage(
                        (
                            nms_function,
                            (polygons, dist_method, "Default", "IOU", 0.5, 0.5),
                        ),
                        interval=0.1,
                        max_usage=True,
                    )
                    end_time = timeit.default_timer()
                    avg_time.append(end_time - start_time)
                    avg_memory.append(mem_usage)

                    # Close distributed if needed
                    if method == "Dask":
                        client.close()
                        time.sleep(5)
                    elif method == "Ray":
                        ray.shutdown()
                        time.sleep(5)

                if (method, avg_overlap) not in time_results:
                    time_results[(method, avg_overlap)] = [
                        sum(avg_time) / len(avg_time)
                    ]
                else:
                    time_results[(method, avg_overlap)].append(
                        sum(avg_time) / len(avg_time)
                    )

                if (method, avg_overlap) not in memory_results:
                    memory_results[(method, avg_overlap)] = [
                        sum(avg_memory) / len(avg_memory)
                    ]
                else:
                    memory_results[(method, avg_overlap)].append(
                        sum(avg_memory) / len(avg_memory)
                    )

    # Plots
    mpl.rcParams["lines.linewidth"] = 3
    plt.style.use("dark_background")
    fig, axis = plt.subplots(1, 2, figsize=(15, 20))

    # Plot time results
    for key, values in time_results.items():
        method, overlap = key
        axis[0].plot(
            num_polygons_list,
            values,
            marker=("*"),
            label=f"Distribution: {method} (Avg Overlap: {overlap})",
        )
    axis[0].set_xlabel("Number of Polygons")
    axis[0].set_ylabel("Elapsed time (seconds) (mean of 5 runs)")
    axis[0].legend()
    axis[0].grid(visible=True)
    axis[0].set_title("Time Benchmark")

    # Plot memory results
    for key, values in memory_results.items():
        method, overlap = key
        axis[1].plot(
            num_polygons_list,
            values,
            marker=("*"),
            label=f"Distribution: {method} (Avg Overlap: {overlap})",
        )
    axis[1].set_xlabel("Number of Polygons")
    axis[1].set_ylabel("Used memory (MB) (mean of 5 runs)")
    axis[1].legend()
    axis[1].grid(visible=True)
    axis[1].set_title("Memory Benchmark")

    # Save plots
    plt.title("Polygon NMS Benchmark Results")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "benchmark_results.png")
    print(f"Saving plots to {save_path} ...")
    plt.savefig(save_path)


if __name__ == "__main__":
    # Parse arguments, get the save directory
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", type=str, default="./", help="Directory to save the results"
    )
    args = parser.parse_args()

    # Get the number of CPUs
    num_cpus = psutil.cpu_count(logical=False)
    print("Number of CPUs:", num_cpus)

    # Get the system memory usage
    mem_usage = psutil.virtual_memory()
    avail_mem = mem_usage.available / (1024 * 1024 * 1024)
    print("Available memory:", round(avail_mem, 2), "GB")

    print(
        f"Benchmarking NMS methods with distributed on {num_cpus} number of workers..."
    )
    benchmark_nms(
        nms_methods=["Not", "Dask", "Ray"], save_dir=args.save_dir, num_workers=num_cpus
    )
