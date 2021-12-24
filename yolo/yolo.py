from typing import Tuple

import numpy as np


def Rect2Box(xyxy: np.ndarray) -> np.ndarray:
    if not isinstance(xyxy, np.ndarray):
        xyxy = np.array(xyxy)
    if xyxy.shape[-1] != 4:
        raise ValueError("xyxy should have a shape of (..., 4)")
    xywh = np.zeros(xyxy.shape)
    xywh[..., 2:4] = xyxy[..., 2:4] - xyxy[..., 0:2]
    xywh[..., 0:2] = (xyxy[..., 2:4] + xyxy[..., 0:2]) / 2
    return xywh


def Box2Rect(xywh: np.ndarray) -> np.ndarray:
    if not isinstance(xywh, np.ndarray):
        xywh = np.array(xywh)
    if xywh.shape[-1] != 4:
        raise ValueError("xywh should have a shape of (..., 4)")
    xyxy = np.zeros(xywh.shape)
    xyxy[..., 0:2] = xywh[..., 0:2] - (xywh[..., 2:4] / 2)
    xyxy[..., 2:4] = xywh[..., 0:2] + (xywh[..., 2:4] / 2)
    return xyxy


def lies_inside(xyxy: np.ndarray, xy: np.ndarray) -> bool:
    if not isinstance(xyxy, np.ndarray):
        xyxy = np.array(xyxy)
    if not isinstance(xy, np.ndarray):
        xy = np.array(xy)
    if xyxy.shape[-1] != 4:
        raise ValueError("xyxy should have a shape of (..., 4)")

    return np.all(np.logical_and(xyxy[..., 0:2] <= xy, xyxy[2:4] >= xy))


def isintersect(xyxy_a: np.ndarray, xyxy_b: np.ndarray) -> bool:
    if not isinstance(xyxy_a, np.ndarray):
        xyxy_a = np.array(xyxy_a)
    if not isinstance(xyxy_b, np.ndarray):
        xyxy_b = np.array(xyxy_b)
    inter_min = np.maximum(xyxy_a[..., 0:2], xyxy_b[..., 0:2])
    inter_max = np.minimum(xyxy_a[..., 2:4], xyxy_b[..., 2:4])
    wh = inter_max - inter_min
    wh[wh < 0] = 0
    area = np.prod(wh, axis=-1, keepdims=True)
    return np.all(area > 0.0)


def intersect_area(xyxy_a: np.ndarray, xyxy_b: np.ndarray) -> np.ndarray:
    if not isinstance(xyxy_a, np.ndarray):
        xyxy_a = np.array(xyxy_a)
    if not isinstance(xyxy_b, np.ndarray):
        xyxy_b = np.array(xyxy_b)
    inter_min = np.maximum(xyxy_a[..., 0:2], xyxy_b[..., 0:2])
    inter_max = np.minimum(xyxy_a[..., 2:4], xyxy_b[..., 2:4])
    wh = inter_max - inter_min
    wh[wh < 0] = 0
    area = np.prod(wh, axis=-1, keepdims=True)
    return area


def union_area(xyxy_a: np.ndarray, xyxy_b: np.ndarray, intersection_area: np.ndarray = None) -> np.ndarray:
    if not isinstance(xyxy_a, np.ndarray):
        xyxy_a = np.array(xyxy_a)
    if not isinstance(xyxy_b, np.ndarray):
        xyxy_b = np.array(xyxy_b)
    wh_a = xyxy_a[..., 2:4] - xyxy_a[..., 0:2]
    wh_b = xyxy_b[..., 2:4] - xyxy_b[..., 0:2]
    area_a = np.prod(wh_a, axis=-1, keepdims=True)
    area_b = np.prod(wh_b, axis=-1, keepdims=True)
    if intersection_area is None:
        intersection_area = intersect_area(xyxy_a, xyxy_b)
    return area_a + area_b - intersection_area


def IoU(xyxy_a: np.ndarray, xyxy_b: np.ndarray) -> np.ndarray:
    intersection = intersect_area(xyxy_a, xyxy_b)
    union = union_area(xyxy_a, xyxy_b, intersection_area=intersection)
    return intersection / union


def PreprocessLabel(
    classes: np.ndarray,
    num_classes: int,
    bounding_boxes: np.ndarray,
    grid: Tuple[int, int] = (7, 7),
) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(classes, np.ndarray):
        classes = np.array(classes, dtype=np.int32)
    if not isinstance(bounding_boxes, np.ndarray):
        bounding_boxes = np.array(bounding_boxes, dtype=np.float32)

    if bounding_boxes.shape[-1] != 4:
        raise ValueError("bounding_boxes should have a shape of (..., 4)")
    if len(grid) != 2:
        raise ValueError("grid should be a tuple of 2 int's")

    labels = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            x_min, y_min = (1 / grid[0]) * i, (1 / grid[1]) * j
            x_max, y_max = x_min + (1 / grid[0]), y_min + (1 / grid[1])
            xyxy = np.array([x_min, y_min, x_max, y_max])
            label = np.zeros(5 + num_classes)

            for x in range(len(bounding_boxes)):
                bounding_box = Rect2Box(bounding_boxes[x])
                if lies_inside(xyxy, bounding_box[..., 0:2]):
                    label[0] = 1
                    label[1:5] = bounding_box
                    label[5 + classes[x]] = 1
            labels.append(label)
    return np.reshape(labels, (*grid, (5 + num_classes)))
