import functools

import numpy as np
import pytest
from shapely.geometry import Polygon

from polygone_nms.nms import (
    apply_distributed_polygon_nms,
    apply_polygon_nms,
    cluster_polygons,
    nms,
)
from polygone_nms.utils import build_rtree, iou


def test_cluster_polygons():
    polygons = [
        (Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), 0, 0.9),
        (Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]), 0, 0.9),
        (Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]), 0, 0.9),
        (Polygon([(0, 0.5), (1, 0.5), (1, 2), (0, 2)]), 0, 0.9),
        (Polygon([(2, 1), (3, 1), (3, 3), (2, 3)]), 0, 0.9),
    ]

    rtree = build_rtree(polygons)

    # Test cluster_polygons function
    clustered_polygons = cluster_polygons(polygons, rtree)
    assert len(clustered_polygons) == 3
    assert sorted(clustered_polygons[0]) == [0, 3]
    assert sorted(clustered_polygons[1]) == [1, 4]
    assert sorted(clustered_polygons[2]) == [2]


@pytest.mark.parametrize(
    "polygons,nms_method,intersection_method,threshold,sigma,expected",
    [
        (
            [
                (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 1, 0.9),
                (Polygon([(1, 0), (3, 0), (3, 2), (1, 2)]), 1, 0.8),
                (Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]), 1, 0.95),
            ],
            "Default",
            iou,
            0.3,
            0.5,
            [0, 2],
        ),
        (
            [
                (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 1, 0.9),
                (Polygon([(1, 0), (3, 0), (3, 2), (1, 2)]), 1, 0.8),
                (Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]), 1, 0.95),
            ],
            "Soft",
            iou,
            0.5,
            0.5,
            [0, 1, 2],
        ),
        (
            [
                (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 1, 0.9),
                (Polygon([(1, 0), (3, 0), (3, 2), (1, 2)]), 1, 0.8),
                (Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]), 1, 0.95),
            ],
            "Soft",
            iou,
            0.7,
            0.5,
            [0, 2],
        ),
        (
            [
                (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 1, 0.9),
                (Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), 2, 0.8),
                (Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]), 1, 0.95),
            ],
            "Class Agnostic",
            iou,
            0.5,
            0.5,
            [0, 1, 2],
        ),
        (
            [
                (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 1, 0.9),
                (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 1, 0.7),
            ],
            "Default",
            iou,
            0.5,
            0.5,
            [0],
        ),
        (
            [
                (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 1, 0.9),
                (Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), 1, 0.8),
                (Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]), 2, 0.85),
                (Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]), 1, 0.7),
                (Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]), 2, 0.95),
                (Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]), 1, 0.6),
            ],
            "Class Agnostic",
            iou,
            0.12,
            0.5,
            [0, 2, 4],
        ),
    ],
)
def test_apply_polygon_nms(
    polygons, nms_method, intersection_method, threshold, sigma, expected
):
    result = apply_polygon_nms(
        polygons, nms_method, intersection_method, threshold, sigma
    )
    assert sorted(result) == expected


def test_fail_apply_polygon_nms():
    polygons = [
        (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 1, 0.9),
        (Polygon([(1, 0), (3, 0), (3, 2), (1, 2)]), 1, 0.8),
        (Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]), 1, 0.95),
    ]
    with pytest.raises(ValueError) as excinfo:
        apply_polygon_nms(polygons, "Unknown", iou, 0.3, 0.5)
    assert "Invalid NMS method: Unknown." in str(excinfo.value)


@pytest.mark.ray_distributed
def test_ray_polygon_nms():
    import ray

    ray.init()

    polygons_first = [
        (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 1, 0.9),
        (Polygon([(1, 0), (3, 0), (3, 2), (1, 2)]), 1, 0.8),
        (Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]), 1, 0.95),
    ]
    polygons_second = [
        (Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]), 1, 0.9),
        (Polygon([(11, 10), (13, 10), (13, 12), (11, 12)]), 1, 0.8),
        (Polygon([(14, 14), (16, 14), (16, 16), (14, 16)]), 1, 0.95),
    ]
    clusters = [polygons_first, polygons_second]
    nms_polygons_futures = [
        ray.remote(apply_distributed_polygon_nms).remote(
            cluster_polygons, "Default", iou, 0.3, 0.5
        )
        for cluster_polygons in clusters
    ]
    nms_polygons = [
        nms_polygon
        for nms_cluster in ray.get(nms_polygons_futures)
        for nms_polygon in nms_cluster
    ]

    assert sorted(nms_polygons) == [0, 0, 2, 2]
    ray.shutdown()


@pytest.mark.dask_distributed
def test_dask_polygon_nms():
    from dask.distributed import Client

    polygons_first = [
        (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 1, 0.9),
        (Polygon([(1, 0), (3, 0), (3, 2), (1, 2)]), 1, 0.8),
        (Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]), 1, 0.95),
    ]
    polygons_second = [
        (Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]), 1, 0.9),
        (Polygon([(11, 10), (13, 10), (13, 12), (11, 12)]), 1, 0.8),
        (Polygon([(14, 14), (16, 14), (16, 16), (14, 16)]), 1, 0.95),
    ]
    clusters = [polygons_first, polygons_second]

    with Client() as client:
        nms_polygon_futures = client.map(
            functools.partial(
                apply_distributed_polygon_nms,
                nms_method="Default",
                intersection_method=iou,
                threshold=0.3,
                sigma=0.5,
            ),
            clusters,
        )
        results = client.gather(nms_polygon_futures)
        nms_polygons = [idx for result in results for idx in result]

    assert sorted(nms_polygons) == [0, 0, 2, 2]


@pytest.mark.parametrize(
    "nms_method,intersection_method,threshold,sigma,expected",
    [
        ("Default", "IOU", 0.3, 0.5, [0, 2, 3, 5]),
        ("Default", "IOS", 0.3, 0.5, [0, 2, 3, 5]),
        ("Default", "Dice", 0.3, 0.5, [0, 2, 3, 5]),
        ("Default", "IOT", 0.3, 0.5, [0, 2, 3, 5]),
        ("Soft", "IOU", 0.7, 0.5, [0, 2, 3, 5]),
        ("Class Agnostic", "IOU", 0.3, 0.5, [0, 2, 3, 5]),
    ],
)
def test_nms(nms_method, intersection_method, threshold, sigma, expected):
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

    results = nms(input_data, None, nms_method, intersection_method, threshold, sigma)
    assert sorted(results) == expected


def test_nms_with_different_input_types():
    input_data = [
        np.array([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.9]),
        np.array([1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0, 1.0, 0.8]),
        np.array([4.0, 4.0, 6.0, 4.0, 6.0, 6.0, 4.0, 6.0, 5.0, 0.95]),
        np.array([10.0, 10.0, 12.0, 10.0, 12.0, 12.0, 10.0, 12.0, 11.0, 0.9]),
        np.array([11.0, 10.0, 13.0, 10.0, 13.0, 12.0, 11.0, 12.0, 11.0, 0.8]),
        np.array([14.0, 14.0, 16.0, 14.0, 16.0, 16.0, 14.0, 16.0, 15.0, 0.95]),
    ]

    results = nms(input_data, None, "Default", "IOU", 0.3, 0.5)
    assert sorted(results) == [0, 2, 3, 5]

    input_data_polygons = [
        (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 1, 0.9),
        (Polygon([(1, 0), (3, 0), (3, 2), (1, 2)]), 1, 0.8),
        (Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]), 1, 0.95),
        (Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]), 1, 0.9),
        (Polygon([(11, 10), (13, 10), (13, 12), (11, 12)]), 1, 0.8),
        (Polygon([(14, 14), (16, 14), (16, 16), (14, 16)]), 1, 0.95),
    ]

    results = nms(input_data_polygons, None, "Default", "IOU", 0.3, 0.5)
    assert sorted(results) == [0, 2, 3, 5]

    input_empty_list = []
    results = nms(input_empty_list, None, "Default", "IOU", 0.3, 0.5)
    assert results == []


def test_fail_nms():
    dump_data = np.array([[0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.9]])

    with pytest.raises(ValueError) as excinfo:
        nms(0, None, "Default", "IOU", 0.3, 0.5)
    assert (
        "Invalid input data type. Expected a list of polygons or a 2D NumPy array."
        in str(excinfo.value)
    )

    with pytest.raises(ValueError) as excinfo:
        nms(dump_data, "Wrong", "Default", "IOU", 0.3, 0.5)

    assert "Invalid distributed computing method: Wrong." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        nms(dump_data, None, "Wrong", "IOU", 0.3, 0.5)

    assert "Invalid NMS method: Wrong." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        nms(dump_data, None, "Default", "Wrong", 0.3, 0.5)

    assert "Invalid intersection method: Wrong." in str(excinfo.value)


@pytest.mark.ray_distributed
def test_ray_nms():
    import ray

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

    results = nms(input_data, "Ray", "Default", "IOU", 0.3, 0.5)
    assert sorted(results) == [0, 2, 3, 5]
    assert not ray.is_initialized()

    ray.init()
    assert ray.is_initialized()
    results = nms(input_data, "Ray", "Default", "IOU", 0.3, 0.5)
    assert sorted(results) == [0, 2, 3, 5]
    assert ray.is_initialized()
    ray.shutdown()
    assert not ray.is_initialized()


@pytest.mark.dask_distributed
def test_dask_nms():
    import dask
    from dask.distributed import Client

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

    results = nms(input_data, "Dask", "Default", "IOU", 0.3, 0.5)
    assert sorted(results) == [0, 2, 3, 5]
    assert not dask.config.get("distributed", {}).get("active", False)

    with Client() as client:
        results = nms(input_data, "Dask", "Default", "IOU", 0.3, 0.5, client=client)
        assert sorted(results) == [0, 2, 3, 5]
    assert not dask.config.get("distributed", {}).get("active", False)
