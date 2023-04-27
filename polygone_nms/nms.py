# Import necessary libraries
import functools
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import rtree
from shapely.geometry import Polygon

# Import package modules
from polygone_nms.utils import (  # , dfs_recursive
    build_rtree,
    create_polygon,
    dfs_iterative,
    dice,
    ios,
    iot,
    iou,
)

INTERSECTION_METHODS = {
    "IOU": iou,
    "IOT": iot,
    "IOS": ios,
    "Dice": dice,
}

VALID_NMS_METHODS = ["Default", "Soft", "Class Agnostic"]
VALID_DISTRIBUTED_METHODS = ["Dask", "Ray"]


def cluster_polygons(
    polygons: List[Tuple[Polygon, float, float]], rtree_index: rtree.index.Index
) -> List[List[Polygon]]:
    """
    Cluster polygons into non-overlapping subregions with R-Tree.
    Used for distributed computing.

    Examples:
        >>> from shapely.geometry import Polygon
        >>> from polygone_nms.utils import build_rtree, cluster_polygons
        >>> polygons = [
        ...     (Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), 0, 0.9),
        ...     (Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]), 0, 0.9),
        ...     (Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]), 0, 0.9),
        ...     (Polygon([(0, 0.5), (1, 0.5), (1, 2), (0, 2)]), 0, 0.9),
        ...     (Polygon([(2, 1), (3, 1), (3, 3), (2, 3)]), 0, 0.9),
        ... ]
        >>> rtree = build_rtree(polygons)
        >>> clustered_polygons = cluster_polygons(polygons, rtree)
        >>> assert len(clustered_polygons) == 3
        >>> assert sorted(clustered_polygons[0]) == [0, 3]
        >>> assert sorted(clustered_polygons[1]) == [1, 4]

    Args:
        polygons (List[Tuple[Polygon, float, float]]): List of polygons,
            where each polygon is a tuple of the polygon, the class label,
            and the confidence score.
        rtree_index (rtree.index.Index): R-Tree index of the polygons.

    Returns:
        List[List[Polygon]]:
            A list of clusters,
            where each cluster is a list of non-overlapping polygons.
    """
    # Create a list of adjacent polygons for each polygon and
    # a list of visited polygons
    adj_list: List[List[int]] = [[] for _ in range(len(polygons))]
    visited = [False] * len(polygons)

    # Iterate over all polygons
    for i, poly_tuple in enumerate(polygons):
        # Get the polygon
        poly = poly_tuple[0]
        # Get all polygons that intersect with the current polygon
        intersecting_polygons = list(rtree_index.intersection(poly.bounds))
        # Remove the current polygon from the list of intersecting polygons
        intersecting_polygons.remove(i)
        # Update the adjacency list
        adj_list[i] = intersecting_polygons

    # Create a list of clusters
    clusters = []
    # Iterate over all polygons
    for i in range(len(polygons)):
        # If the current polygon has not been visited
        if not visited[i]:
            # Perform a depth-first search to find all connected components
            # Due to RecursionError: maximum recursion depth exceeded
            # connected_component = dfs_recursive(i, visited, adj_list)
            connected_component = dfs_iterative(i, visited, adj_list)
            # # Create a cluster from the connected component
            # cluster = [polygons[j] for j in connected_component]
            # # Append the cluster to the list of clusters
            # Append the connected component (cluster) to the list of clusters
            clusters.append(connected_component)

    return clusters


def apply_polygon_nms(
    polygons: List[Tuple[Polygon, float, float]],
    nms_method: str,
    intersection_method: Callable,
    threshold: float = 0.5,
    sigma: float = 0.5,
) -> List[int]:
    """
    Apply Non-Maximum Suppression (NMS) to a list of predicted polygons.

    Raises:
        ValueError: If the NMS method is invalid.

    Args:
        polygons (List[Tuple[Polygon, float, float]]):
            List of polygons, where each polygon is a tuple of the polygon
            represented by shapely.geometry.Polygon object, the class label,
            and the confidence score.
        nms_method (str): The NMS method to use, one of
            ("Default", "Soft", "Class Agnostic").
        intersection_method (Callable): The method to compute intersections.
        threshold (float, optional): The threshold for the NMS method. Defaults to 0.5.
        sigma (float, optional): The sigma for the Soft NMS method. Defaults to 0.5.

    Returns:
        List[int]: A list of kept polygon indices.
    """
    # List of kept polygons
    kept_polygons = []

    # Sort polygons by confidence score
    confidences = np.array([poly[2] for poly in polygons])
    sorted_indices = np.argsort(confidences)[::-1]

    # Iterate over all sorted polygons by confidence score
    while sorted_indices.size > 0:
        # Get the index of the current polygon and the respective polygon and label
        i = sorted_indices[0]
        current_polygon = polygons[i][0]
        current_class = polygons[i][1]

        # Add the current polygon to the NMS kept polygons list
        kept_polygons.append(i)

        # Calculate the intersection between the current polygon and
        # all remaining polygons
        intersections = [
            intersection_method(current_polygon, polygons[j][0])
            for j in sorted_indices[1:]
        ]

        # If the NMS method is "Default", "Soft" or "Class Agnostic"
        if nms_method == "Default":
            # Remove all remaining polygons that have an intersection with
            # the current polygon greater than the threshold
            remaining_indices = [
                idx
                for idx, intersection in zip(sorted_indices[1:], intersections)
                if polygons[idx][1] != current_class or intersection <= threshold
            ]
        elif nms_method == "Soft":
            # Update the confidence scores of the remaining polygons
            for j, intersection in zip(sorted_indices[1:], intersections):
                # Calculate the weight for the current polygon equal to
                # e^(-intersection^2 / sigma)
                weight = np.exp(-(intersection**2) / sigma)
                confidences[j] *= weight

            # Discard polygons with confidence below the threshold
            remaining_indices = [
                idx for idx in sorted_indices[1:] if confidences[idx] >= threshold
            ]
        elif nms_method == "Class Agnostic":
            # Remove all remaining polygons that have an intersection with
            # the current polygon greater than the threshold and
            # have the same class label
            remaining_indices = [
                idx
                for idx, intersection in zip(sorted_indices[1:], intersections)
                if intersection <= threshold
            ]
        else:
            raise ValueError(
                (
                    f"Invalid NMS method: {nms_method}. "
                    f"Allowed methods are: {VALID_NMS_METHODS}"
                )
            )

        # Update the list of remaining polygon indices
        sorted_indices = np.array(remaining_indices)

    return kept_polygons


def apply_distributed_polygon_nms(
    polygons: List[Tuple[Polygon, float, float]],
    nms_method: str,
    intersection_method: Callable,
    threshold: float = 0.5,
    sigma: float = 0.5,
) -> List[int]:
    """
    Distributed version of `apply_polygon_nms`.

    Args:
        polygons (List[Tuple[Polygon, float, float]]):
            List of polygons, where each polygon is a tuple of the polygon
            represented by shapely.geometry.Polygon object, the class label,
            and the confidence score.
        nms_method (str): The NMS method to use, one of
            ("Default", "Soft", "Class Agnostic").
        intersection_method (Callable): The method to compute intersections.
        threshold (float, optional): The threshold for the NMS method. Defaults to 0.5.
        sigma (float, optional): The sigma for the Soft NMS method. Defaults to 0.5.

    Returns:
        List[int]: A list of kept polygon indices.
    """
    return apply_polygon_nms(
        polygons, nms_method, intersection_method, threshold, sigma
    )


def nms(
    input_data: Union[np.ndarray, Tuple[Polygon, float, float]],
    distributed: Optional[str] = None,
    nms_method: str = "Default",
    intersection_method: str = "IOU",
    threshold: float = 0.5,
    sigma: float = 0.5,
    **kwargs,
) -> List[int]:
    """
    Apply Non-Maximum Suppression (NMS) to a set of polygons.
    Method works with distributed computing for efficient processing and clustering.

    Examples:
        >>> import numpy as np
        >>> from polygone.nms import nms
        >>> input_data = np.array(
        ... [
        ...     [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.9],
        ...     [1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0, 1.0, 0.8],
        ...     [4.0, 4.0, 6.0, 4.0, 6.0, 6.0, 4.0, 6.0, 5.0, 0.95],
        ...     [10.0, 10.0, 12.0, 10.0, 12.0, 12.0, 10.0, 12.0, 11.0, 0.9],
        ...     [11.0, 10.0, 13.0, 10.0, 13.0, 12.0, 11.0, 12.0, 11.0, 0.8],
        ...     [14.0, 14.0, 16.0, 14.0, 16.0, 16.0, 14.0, 16.0, 15.0, 0.95],
        ... ])
        >>> results = nms(input_data, None, "Default", "IOU", 0.3, 0.5)
        >>> assert sorted(results) == [0, 2, 3, 5]

    results = nms(input_data, None, nms_method, intersection_method, threshold, sigma)
    assert sorted(results) == expected

    Args:
        input_data (Union[np.ndarray, Tuple[Polygon, float, float]]):
            List of tuples of Polygon, class, score or numpy array of polygons.
            Each polygon in the numpy array is represented by a 1D array of n % 2
            coordinates (x1, y1, x2, y2, .., x(n-1), y(n-1), class, score).
        distributed (Optional[str], optional):
            The distributed computing method to use,
            one of (None, "Ray", "Dask").. Defaults to None.
        nms_method (str, optional):
            The NMS method to use, one of ("Default", "Soft", "Class Agnostic").
            Defaults to "Default".
        intersection_method (str, optional):
            The method to compute intersections, one of ("IOU", "IOS", "Dice", "IOT").
            Defaults to "IOU".
        threshold (float, optional):
            The threshold for the NMS(intersection) method. Defaults to 0.5.
        sigma (float, optional):
            The sigma for the Soft NMS method. Defaults to 0.5.
        **kwargs: Additional arguments for the NMS method.
            Any keyword arguments for the distributed computing should be passed here.

    Returns:
        List[int]: List of indices of the kept polygons.
    """
    # Check if the input data is a list of polygons or a 2D NumPy array
    polygons = []
    if isinstance(input_data, list):
        if len(input_data) == 0:
            return []
        elif isinstance(input_data[0], tuple):
            polygons = input_data
        else:
            input_data = np.array(input_data)
    elif not isinstance(input_data, np.ndarray):
        raise ValueError(
            "Invalid input data type. Expected a list of polygons or a 2D NumPy array."
        )

    # Check distributed computing method
    if distributed is not None:
        if distributed not in VALID_DISTRIBUTED_METHODS:
            raise ValueError(
                (
                    f"Invalid distributed computing method: {distributed}. "
                    f"Allowed methods are: {VALID_DISTRIBUTED_METHODS}"
                )
            )

    # Check NMS method
    if nms_method not in VALID_NMS_METHODS:
        raise ValueError(
            (
                f"Invalid NMS method: {nms_method}. "
                f"Allowed methods are: {VALID_NMS_METHODS}"
            )
        )

    # Check intersection method
    if intersection_method not in list(INTERSECTION_METHODS.keys()):
        raise ValueError(
            (
                f"Invalid intersection method: {intersection_method}. "
                f"Allowed methods are: {list(INTERSECTION_METHODS.keys())}"
            )
        )

    # Get the intersection function
    intersection_func = INTERSECTION_METHODS[intersection_method]

    # Convert input data to Shapely Polygons and store class and confidence
    if len(polygons) == 0:
        for row in input_data:
            polygon_coords = row[:-2]
            class_label = row[-2]
            confidence = row[-1]
            polygon = create_polygon(polygon_coords)
            polygons.append((polygon, class_label, confidence))

    # Build R-Tree index
    rtree = build_rtree(polygons)

    # Cluster polygons into non-overlapping subregions
    clusters = cluster_polygons(polygons, rtree)
    polygon_clusters = [[polygons[idx] for idx in cluster] for cluster in clusters]

    # Apply NMS to the clustered polygons
    if distributed == "Ray":
        # Import Ray
        import ray

        # Check if Ray is initialized
        if ray.is_initialized():
            initialize_ray = False
        else:
            initialize_ray = True
            ray.init()  # Initialize Ray

        # Scatter the polygon clusters to the Ray cluster
        polygon_futures = [
            ray.put(polygon_cluster) for polygon_cluster in polygon_clusters
        ]

        # Apply NMS to each cluster
        nms_polygons_futures = [
            ray.remote(apply_distributed_polygon_nms).remote(
                polygon_future, nms_method, intersection_func, threshold, sigma
            )
            for polygon_future in polygon_futures
        ]
        nms_polygons = ray.get(nms_polygons_futures)

        # Shutdown Ray if it was initialized in this function
        if initialize_ray:
            ray.shutdown()  # Shutdown Ray
    elif distributed == "Dask":
        # Use an existing Dask client if provided, or create a new one
        client = kwargs.get("client", None)
        if client is None:
            from dask.distributed import Client

            initialize_dask = True
            client = Client()
        else:
            initialize_dask = False

        # Scatter the polygon clusters to the Dask cluster
        polygon_futures = client.scatter(polygon_clusters)

        # Apply NMS to each cluster and gather the results
        nms_polygon_futures = client.map(
            functools.partial(
                apply_distributed_polygon_nms,
                nms_method=nms_method,
                intersection_method=intersection_func,
                threshold=threshold,
                sigma=sigma,
            ),
            polygon_futures,
        )
        nms_polygons = client.gather(nms_polygon_futures)
        # Shutdown Dask if it was initialized in this function
        if initialize_dask:
            client.close()  # Shutdown Dask
    else:
        # Apply NMS to each cluster
        nms_polygons = [
            apply_distributed_polygon_nms(
                polygon_cluster, nms_method, intersection_func, threshold, sigma
            )
            for polygon_cluster in polygon_clusters
        ]

    # Combine the kept polygon indices from each cluster and
    # get the keep indices from each cluster
    keep_indices = []
    for cluster, cluster_nms in zip(clusters, nms_polygons):
        for idx in cluster_nms:
            keep_indices.append(cluster[idx])

    return keep_indices
