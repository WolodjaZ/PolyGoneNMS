# PolyGoneNMS

PolyGoneNMS is a library for efficient and distributed polygon Non-Maximum Suppression (NMS) in Python. It supports various NMS methods, intersection calculations, and can handle large numbers of polygons in 1D, 2D, and 3D spaces. PolyGoneNMS uses R-tree data structures and shapely polygon objects for optimal performance.

## Benchmark Plots

![Benchmark results](assets/benchmark_results.png)

## Features

- Efficient polygon NMS for large numbers of polygons.
- Support for various NMS methods: Default, Soft, and Class Agnostic.
- Support for different intersection methods: IOU, IOS, and Dice.
- R-tree data structure for efficient spatial indexing and querying.
- Distributed processing support using Ray and Dask.
- Comprehensive documentation and examples.

## Installation

You can install PolyGoneNMS using pip:

```bash
pip install polygone-nms
```

## Quickstart

```python
import numpy as np
from polygone_nms import polygone_nms

# Example input data
data = np.array([
    [0, 0, 1, 1, 0, 1, 0, 0, 1, 0.9],
    [0.5, 0.5, 1.5, 1.5, 0.5, 1.5, 0, 0, 1, 0.8],
])

# Apply NMS
results = nms(data, distributed=None, nms_method="Default", intersection_method="IOU")

print("Filtered indices:", results)

# Filtered data
print("Filtered data:")
print(data[results])
```

For a more detailed guide on using PolyGoneNMS, please see the [Quickstart](https://wolodjaz.github.io/PolyGoneNMS/0.1.6/quickstart/) in the documentation.

## Documentation

Detailed documentation is available at:
[![Docs](https://img.shields.io/badge/Docs-mkdocs-blue?style=flat)](https://wolodjaz.github.io/PolyGoneNMS)


## Contributing

We welcome contributions to the project! Please follow the usual GitHub process for submitting issues or pull requests.

## License

This project is licensed under the [MIT License](https://fossa.com/blog/open-source-licenses-101-mit-license/).
