#!/bin/sh

version_changed=false
benchmark_changed=false

if git diff --cached --name-only | grep -q VERSION; then
  version_changed=true
fi

if git diff --cached --name-only | grep -q assets/benchmark_results.png; then
  benchmark_changed=true
fi

if [ "$version_changed" = true ] && [ "$benchmark_changed" = false ]; then
  echo "ERROR: The VERSION file has changed, but the assets/benchmark_results.png file has not. Please update the benchmark results before committing."
  exit 1
elif [ "$version_changed" = true ]; then
  echo "WARNING: The VERSION file has changed. Remember to run benchmark.py if applicable."
fi

exit 0
