import os

import numpy as np

import polygone_nms


def get_version():
    with open(os.path.join(os.path.dirname(__file__), "..", "VERSION")) as f:
        return f.read().strip()


def test_polygone_nms():
    assert polygone_nms.__version__ == get_version()


def test_polygone_nms_nms():
    input_data = np.array(
        [
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.9],
            [1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0, 1.0, 0.8],
            [4.0, 4.0, 6.0, 4.0, 6.0, 6.0, 4.0, 6.0, 5.0, 0.95],
            [10.0, 10.0, 12.0, 10.0, 12.0, 12.0, 10.0, 12.0, 11.0, 0.9],
            [11.0, 10.0, 13.0, 10.0, 13.0, 12.0, 11.0, 12.0, 11.0, 0.8],
            [14.0, 14.0, 16.0, 14.0, 16.0, 16.0, 14.0, 16.0, 15.0, 0.95],
        ]
    )

    results = polygone_nms.nms(input_data, None, "Default", "IOU", 0.3, 0.5)
    assert sorted(results) == [0, 2, 3, 5]
