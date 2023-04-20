import os

import polygone_nms


def get_version():
    with open(os.path.join(os.path.dirname(__file__), "..", "VERSION")) as f:
        return f.read().strip()


def test_polygone_nms():
    assert polygone_nms.__version__ == get_version()
