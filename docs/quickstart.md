# Quickstart

This guide will walk you through the process of installing and using the PolyGoneNMS library. By the end, you will have a basic understanding of how to apply Non-Maximum Suppression on polygon data using various methods and settings.

## Installation

You can install the PolyGoneNMS library using `pip`. Run the following command in your terminal or command prompt:

```bash
pip install polygone-nms
```

## Basic Usage

To use the PolyGoneNMS library, you'll need to import the polygone_nms function:

```python
from polygone_nms import polygone_nms
```

Next, prepare your input data as a NumPy or PyTorch array in the following format: `[x1, y1, x2, y2, ..., class, confidence]`. Here's a sample input data:

```python
import numpy as np

data = np.array([
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 0.9],
    [1, 1, 2, 2, 1, 1, 2, 2, 0, 0.8],
    [2, 2, 3, 3, 2, 2, 3, 3, 1, 0.7],
])
```

Now you can apply the `polygone_nms` function on your input data:

```python
result = polygone_nms(data)
```

The `result` variable will contain the filtered polygons after applying the Non-Maximum Suppression.

## Customizing PolyGoneNMS

PolyGoneNMS allows you to customize its behavior using various parameters. Here's an example of how to use Soft NMS with the Dice coefficient:

```python
result = polygone_nms(
    data,
    nms_method="Soft",
    intersection_method="Dice",
)
```

You can also enable distributed processing using Ray or Dask:

```python
result = polygone_nms(
    data,
    distributed="Ray",
)
```

For more information on the available parameters and their usage, refer to the [API Reference](api_reference.md) page.

## Example

Here's a complete example that demonstrates how to use the PolyGoneNMS library:

```python
import numpy as np
from polygone_nms import nms

# Sample input data
data = np.array([
    [0, 0, 1, 1, 0, 1, 0, 0, 1, 0.9],
    [0.5, 0.5, 1.5, 1.5, 0.5, 1.5, 0, 0, 1, 0.8],
])

# Apply PolyGoneNMS with custom settings
result = nms(
    data,
    nms_method="Soft",
    intersection_method="Dice",
#    distributed="Ray",
)

# Print the filtered polygons
print(data[result])
```

This example uses Soft NMS with the Dice coefficient and Ray for distributed processing. The output will be the filtered polygons after applying the Non-Maximum Suppression.
