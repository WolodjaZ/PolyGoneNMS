import numpy as np
import pytest
import rtree
from shapely.geometry import Polygon

from polygone_nms.utils import build_rtree, create_polygon, dfs, dice, ios, iot, iou


def bbox_to_polygon(bbox):
    return Polygon(
        [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])]
    )


@pytest.mark.parametrize(
    "polygon_list",
    [
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [2.0, 0.0, 3.0, 0.0, 3.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        [1.0, 2.0, 2.0, 2.0, 2.0, 3.0, -1.0, -1.0],
    ],
)
def test_create_polygone(polygon_list):
    polygon = create_polygon(np.array(polygon_list))
    assert polygon is not None
    assert isinstance(polygon, Polygon)
    assert polygon.is_valid


def test_build_rtree():
    polygons = [
        (Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), 0, 0.9),
        (Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]), 1, 0.8),
        (Polygon([(1, 2), (2, 2), (2, 3), (1, 3)]), 2, 0.7),
    ]

    rtree_index = build_rtree(polygons)
    assert isinstance(rtree_index, rtree.index.Index)

    # Test that the R-tree index returns the correct intersecting polygons
    query_bounds = (0.5, 0.5, 1.5, 1.5)
    intersecting_polygons = list(rtree_index.intersection(query_bounds))
    for i in range(len(polygons)):
        if i in intersecting_polygons:
            assert polygons[i][0].intersects(bbox_to_polygon(query_bounds))
        else:
            assert not polygons[i][0].intersects(bbox_to_polygon(query_bounds))

    query_bounds = (1.5, 1.5, 2.5, 2.5)
    intersecting_polygons = list(rtree_index.intersection(query_bounds))
    for i in range(len(polygons)):
        if i in intersecting_polygons:
            assert polygons[i][0].intersects(bbox_to_polygon(query_bounds))
        else:
            assert not polygons[i][0].intersects(bbox_to_polygon(query_bounds))


def test_dfs():
    # Test graph structure:
    # 0 -- 1 -- 2
    #      |
    #      3
    adj_list = [[1], [0, 2, 3], [1], [1]]  # Node 0  # Node 1  # Node 2  # Node 3

    visited = [False] * len(adj_list)
    starting_node = 0
    connected_component = dfs(starting_node, visited, adj_list)
    assert connected_component == [0, 1, 2, 3]

    # Test disconnected graph structure:
    # 0 -- 1    2 -- 3
    adj_list = [[1], [0], [3], [2]]  # Node 0  # Node 1  # Node 2  # Node 3

    visited = [False] * len(adj_list)
    starting_node = 0
    connected_component = dfs(starting_node, visited, adj_list)
    assert connected_component == [0, 1]

    starting_node = 2
    connected_component = dfs(starting_node, visited, adj_list)
    assert connected_component == [2, 3]


@pytest.mark.parametrize(
    "poly1_list, poly2_list, expected",
    [
        (
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)],
            0.3,
        ),
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [(2, 0), (3, 0), (3, 1), (2, 1)], 0.0),
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (1, 0), (1, 1), (0, 1)], 1.0),
    ],
)
def test_iou(poly1_list, poly2_list, expected):
    poly1 = Polygon(poly1_list)
    poly2 = Polygon(poly2_list)
    assert round(iou(poly1, poly2), 1) == expected


@pytest.mark.parametrize(
    "poly1_list, poly2_list, expected",
    [
        (
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)],
            0.5,
        ),
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [(2, 0), (3, 0), (3, 1), (2, 1)], 0.0),
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [(-1, -1), (2, -1), (2, 2), (-1, 2)], 1.0),
    ],
)
def test_ios(poly1_list, poly2_list, expected):
    poly1 = Polygon(poly1_list)
    poly2 = Polygon(poly2_list)
    assert round(ios(poly1, poly2), 1) == expected


@pytest.mark.parametrize(
    "poly1_list, poly2_list, expected",
    [
        (
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)],
            0.5,
        ),
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [(2, 0), (3, 0), (3, 1), (2, 1)], 0.0),
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (1, 0), (1, 1), (0, 1)], 1.0),
    ],
)
def test_dice(poly1_list, poly2_list, expected):
    poly1 = Polygon(poly1_list)
    poly2 = Polygon(poly2_list)
    assert round(dice(poly1, poly2), 1) == expected


@pytest.mark.parametrize(
    "poly1_list, poly2_list, expected",
    [
        (
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(0.5, 0), (2.5, 0), (2.5, 1), (0.5, 1)],
            0.5,
        ),
        (
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)],
            0.5,
        ),
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [(2, 0), (3, 0), (3, 1), (2, 1)], 0.0),
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (1, 0), (1, 1), (0, 1)], 1.0),
    ],
)
def test_iot(poly1_list, poly2_list, expected):
    poly1 = Polygon(poly1_list)
    poly2 = Polygon(poly2_list)
    assert round(iot(poly1, poly2), 1) == expected
