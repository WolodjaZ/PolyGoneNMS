import subprocess
import sys


def main():
    version_changed = False
    benchmark_changed = False

    output = subprocess.check_output(["git", "diff", "--cached", "--name-only"]).decode(
        "utf-8"
    )

    if "VERSION" in output:
        version_changed = True

    if "assets/benchmark_results.png" in output:
        benchmark_changed = True

    if version_changed and not benchmark_changed:
        print(
            "ERROR: The VERSION file has changed, "
            "but the assets/benchmark_results.png file has not. "
            "Please update the benchmark results before committing."
        )
        sys.exit(1)
    elif version_changed:
        print(
            "WARNING: The VERSION file has changed. "
            "Remember to run benchmark.py if applicable."
        )


if __name__ == "__main__":
    main()
