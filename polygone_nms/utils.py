from typing import List, Tuple

import numpy as np
import rtree
from rtree import index
from shapely.geometry import Polygon


def bbox_to_polygon_array(coords: np.ndarray) -> np.ndarray:
    """
    Convert bbox [xmin, ymin, xmax, ymax] to polygon format
    [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]

    Args:
        coords (np.ndarray): bbox coordinates
    Returns:
        np.ndarray: polygon coordinates
    """
    return np.array(
        [
            coords[0],
            coords[1],
            coords[2],
            coords[1],
            coords[2],
            coords[3],
            coords[0],
            coords[3],
        ]
    )


def create_polygon(coords: np.ndarray, none_value: float = -1.0) -> Polygon:
    """
    Create a Shapely Polygon from a numpy array row of coordinates.
    If the number of coordinates is odd, an error is raised.
    If any of the coordinates is of `none_value`, it is ignored.

    Examples:
        >>> import numpy as np
        >>> from polygone_nms.utils import create_polygon
        >>> coords = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        >>> poly = create_polygon(coords)
        >>> poly.is_valid
        True
        >>> poly.area
        1.0
        >>> coords = np.array([2.0, 0.0, 3.0, 0.0, 3.0, 1.0, 2.0, 1.0, 2.0, 0.0])
        >>> poly = create_polygon(coords)
        >>> poly.is_valid
        True
        >>> poly.area
        1.0
        >>> coords = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 3.0, -1.0, -1.0])
        >>> poly = create_polygon(coords)
        >>> poly.is_valid
        True
        >>> poly.area
        0.5
        >>> coords = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 3.0, -1.0, -1.0, 1.0, 2.0])
        >>> poly = create_polygon(coords)
        Traceback (most recent call last):
            ...
        ValueError: The number of coordinates must be even.

    Raises:
        ValueError: If the number of coordinates is odd.

    Args:
        coords (np.ndarray):
            A numpy array of coordinates for a shapely.geometry.Polygon.
        none_value (float, optional): A value to be ignored. Defaults to -1.0.

    Returns:
        Polygon: A Shapely Polygon.
    """
    assert len(coords.shape) == 1, "The coordinates must be a 1D array."
    if coords.shape[0] % 2 != 0:
        raise ValueError("The number of coordinates must be even.")
    assert coords.shape[0] % 2 == 0, "The number of coordinates must be even."

    if coords.shape[0] == 4:
        coords = bbox_to_polygon_array(coords)

    points = [
        (coords[i], coords[i + 1])
        for i in range(0, len(coords), 2)
        if coords[i] != none_value and coords[i + 1] != none_value
    ]
    return Polygon(points)


def build_rtree(polygons: List[Tuple[Polygon, float, float]]) -> rtree.index.Index:
    """
    Build an R-tree index from a list of tuples having polygon, class and confidence.

    The R-tree index is used to perform spatial queries on the input polygons.
    The input polygons are represented as Shapely Polygons.

    The R-tree index is built using the rtree library.
    More information about the R-tree library can be found here:
    https://pypi.org/project/rtree/

    Examples:
        >>> from shapely.geometry import Polygon
        >>> from polygone_nms.utils import build_rtree
        >>> p1 = (Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), 0, 0.9)
        >>> p2 = (Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]), 1, 0.8)
        >>> p3 = (Polygon([(1, 2), (2, 2), (2, 3), (1, 3)]), 1, 0.7)
        >>> rtree_index = build_rtree([p1, p2, p3])
        >>> query_bounds = (0.5, 0.5, 1.5, 1.5)
        >>> list(rtree_index.intersection(query_bounds))
        [0]
        >>> query_bounds = (1.5, 1.5, 2.5, 2.5)
        >>> list(rtree_index.intersection(query_bounds))
        [2]

    Args:
        polygons (List[Tuple[Polygon, float, float]]):
            A list of tuples having polygon, class and confidence.

    Returns:
        rtree.index.Index: An R-tree index containing the input polygons.
    """
    # Create an R-tree index
    rtree_idx = index.Index()

    # Iterate over the polygons and add them to the R-tree index
    for idx, row in enumerate(polygons):
        # Retrieve polygon, class, and confidence
        polygon, cls, conf = row

        # Calculate the bounding box of the polygon
        bbox = polygon.bounds  # Returns (minx, miny, maxx, maxy)

        # Insert the bounding box into the R-tree
        rtree_idx.insert(
            idx, bbox, obj={"polygon": polygon, "class": cls, "confidence": conf}
        )

    return rtree_idx


def dfs_recursive(
    node: int, visited: List[bool], adj_list: List[List[int]]
) -> List[int]:
    """
    Perform a depth-first search on an adjacency list (graph) in a recursive manner.

    Examples:
        >>> from polygone_nms.utils import dfs
        >>> adj_list = [[1], [0, 2, 3], [1], [1]]
        >>> visited = [False] * len(adj_list)
        >>> dfs(0, visited, adj_list)
        [0, 1, 2, 3]

    Args:
        node (int): The starting node for the DFS traversal.
        visited (List[bool]):
            list of booleans indicating whether a node has been visited.
        adj_list (List[List[int]]): The adjacency list representing the graph.

    Returns:
        List[int]:
            A list of nodes in the connected component found by the DFS traversal.
    """
    # Mark the node as visited
    visited[node] = True

    # Add the node to the list
    connected_component = [node]

    # Iterate over the neighbors of the node
    for neighbor in adj_list[node]:
        # If the neighbor has not been visited, perform a depth-first search on it
        if not visited[neighbor]:
            # Extend the connected component with the connected component
            # found by the DFS traversal pf the neighbor
            connected_component.extend(dfs_recursive(neighbor, visited, adj_list))

    return connected_component


def dfs_iterative(
    node: int, visited: List[bool], adj_list: List[List[int]]
) -> List[int]:
    """
    Perform a depth-first search on an adjacency list (graph) in a iterative manner.

    Examples:
        >>> from polygone_nms.utils import dfs
        >>> adj_list = [[1], [0, 2, 3], [1], [1]]
        >>> visited = [False] * len(adj_list)
        >>> dfs(0, visited, adj_list)
        [0, 1, 2, 3]

    Args:
        node (int): The starting node for the DFS traversal.
        visited (List[bool]):
            list of booleans indicating whether a node has been visited.
        adj_list (List[List[int]]): The adjacency list representing the graph.

    Returns:
        List[int]:
            A list of nodes in the connected component found by the DFS traversal.
    """
    # Create a list to store the connected component
    connected_component = []

    # Create a stack
    stack = [node]

    # Iterate over the stack
    while stack:
        # Pop the last element from the stack
        curr_node = stack.pop()
        # If the node has not been visited, mark it as visited and
        # add it to the connected component
        if not visited[curr_node]:
            visited[curr_node] = True
            connected_component.append(curr_node)
            # Add the neighbors of the current node to the stack if
            # they have not been visited
            stack.extend(
                reversed(
                    [
                        neighbor
                        for neighbor in adj_list[curr_node]
                        if not visited[neighbor]
                    ]
                )
            )

    return connected_component


def iou(poly1: Polygon, poly2: Polygon) -> float:
    """
    Compute the intersection over union (IoU) between two Shapely Polygons.

    IoU is a popular metric for evaluating the quality of
    object detections and segmentations. It measures the ratio of the area of overlap
    between two regions (e.g., predicted and ground truth bounding boxes)
    to the area of their union.

    IOU = Area of intersection / Area of union
       ~= Area of intersection / (Area of poly1 + Area of poly2 - Area of intersection)

    An IoU score of 1 means that the predicted and ground truth regions
    perfectly overlap, while a score of 0 means that there's no overlap at all.

    Notes:
        In object detection and segmentation tasks,
        a higher IoU threshold indicates a stricter evaluation criterion.

    Examples:
        >>> from shapely.geometry import Polygon
        >>> from polygone_nms.utils import iou
        >>> poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> poly2 = Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
        >>> iou(poly1, poly2)
        0.3333333333333333

    Args:
        poly1 (Polygon):
            The first polygon represented by a shapely.geometry.Polygon object.
        poly2 (Polygon):
            The second polygon represented by a shapely.geometry.Polygon object.

    Returns:
        float: The IoU value between the two Shapely Polygons.
    """
    # Calculate the intersection area
    intersection_area = poly1.intersection(poly2).area

    # Calculate the union
    union_area = poly1.area + poly2.area - intersection_area
    # union_area = poly1.union(poly2).area tested and it's slower

    # If the union area is 0, return 0
    if union_area == 0:
        return 0

    # Calculate the IoU
    return intersection_area / union_area


def ios(poly1: Polygon, poly2: Polygon) -> float:
    """
    Compute the intersection over smaller (IoS) between two Shapely Polygons.

    IoS is another overlap metric that measures the ratio of the area of
    intersection between two regions to the area of the smaller region.

    IoS = (Area of Intersection) / (Area of the Smaller Region)

    An IoU score of 1 means that the predicted and ground truth regions
    perfectly overlap, while a score of 0 means that there's no overlap at all.

    Notes:
        Unlike IoU, IoS is more sensitive to the size of the regions being compared.
        In certain scenarios, using IoS can help to better evaluate the quality
        of detections, especially when dealing with objects of varying sizes.

    Examples:
        >>> from shapely.geometry import Polygon
        >>> from polygone_nms.utils import ios
        >>> poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> poly2 = Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
        >>> ios(poly1, poly2)
        0.5

    Args:
        poly1 (Polygon):
            The first polygon represented by a shapely.geometry.Polygon object.
        poly2 (Polygon):
            The second polygon represented by a shapely.geometry.Polygon object.
    Returns:
        float: The IoS value between the two Shapely Polygons.
    """
    # Calculate the intersection area
    intersection_area = poly1.intersection(poly2).area

    # Calculate the area of the smaller polygon
    smaller_area = min(poly1.area, poly2.area)

    # If the smaller area is 0, return 0
    if smaller_area == 0:
        return 0

    # Calculate the IoS
    return intersection_area / smaller_area


def dice(poly1: Polygon, poly2: Polygon) -> float:
    """
    Compute the dice coefficient between two Shapely Polygons.

    The Dice coefficient is a similarity measure used in image segmentation tasks,
    particularly for comparing binary segmentation masks.
    It is defined as the ratio of twice the area of intersection
    between the predicted and ground truth masks to the sum of their areas.

    Dice=(2 * Area of Intersection) / (Area of First Polygon + Area of Second Polygon)

    The Dice coefficient ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes:
        It is more sensitive to the size of the regions than IoU and is
        particularly useful for evaluating the performance of segmentation algorithms
        in cases where the regions of interest have varying sizes and shapes.

    Examples:
        >>> from shapely.geometry import Polygon
        >>> from polygone_nms.utils import dice
        >>> poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> poly2 = Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
        >>> dice(poly1, poly2)
        0.5

    Args:
        poly1 (Polygon):
            The first polygon represented by a shapely.geometry.Polygon object.
        poly2 (Polygon):
            The second polygon represented by a shapely.geometry.Polygon object.
    Returns:
        float: The Dice value between the two Shapely Polygons.
    """
    # Calculate the intersection
    intersection = poly1.intersection(poly2)

    # Calculate the sum area
    sum_area = poly1.area + poly2.area

    # If the sum area is 0, than Polygons are empty in future are information #TODO
    if sum_area == 0:
        return 0

    # Calculate the Dice coefficient
    return 2 * intersection.area / sum_area


def iot(target: Polygon, compared: Polygon) -> float:
    """
    Compute the intersection over target (IoT) between two Shapely Polygons.

    IoT is another overlap metric that measures the ratio of the area of
    intersection between the Target and Compared regions to the area of
    the Target region.

    IoT = (Area of Intersection) / (Area of the Target Region)

    An IoT score of 1 means that the compared region perfectly overlaps
    the target region, while a score of 0 means that there's no overlap at all.

    Notes:
        This is a testing metrics designed for NMS algorithm.
        My intuition is that if used
        with NMS algorithm it can result in a better performance.

    the predicted and ground truth regions perfectly overlap,
    while a score of 0 means that there's no overlap at all.

    Examples:
        >>> from shapely.geometry import Polygon
        >>> from polygone_nms.utils import iot
        >>> poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> poly2 = Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
        >>> iot(poly1, poly2)
        0.5

    Args:
        target (Polygon):
            The target polygon represented by a shapely.geometry.Polygon object.
        compared (Polygon):
            The second polygon represented by a shapely.geometry.Polygon object.

    Returns:
        float: The IOT value between target and compared Shapely Polygons.
    """
    # Calculate the intersection area
    intersection_area = target.intersection(compared).area

    # Calculate the area of the target polygon
    target_area = target.area

    # If the target area is 0, than Polygons are empty in future are information #TODO
    if target_area == 0:
        return 0

    # Calculate the IoT
    return intersection_area / target_area
