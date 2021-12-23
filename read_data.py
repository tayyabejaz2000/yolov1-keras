from collections import defaultdict
from os import listdir
from random import shuffle
from typing import Dict, List, Tuple, Union
from xml.etree.cElementTree import Element, ElementTree

import cv2 as cv
import numpy as np

from yolo import yolo

PASCAL_VOC_CLASSES = [
    "person",
    "bird", "cat", "cow", "dog", "horse", "sheep",
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor",
]


def XMLtoDict(tree: Element) -> Dict:
    """
    Utility Function to convert Pascal VOC (XML) format to python Dict

    `tree: xml.etree.cElementTree.Element`: The input XML Tree

    returns:
    `Dict` Python Dictionary with all xml elements as key-value pair
    """
    dictionary = {tree.tag: {} if tree.attrib else None}
    children = list(tree)
    if children:
        dd = defaultdict(list)
        for dc in map(XMLtoDict, children):
            for k, v in dc.items():
                dd[k].append(v)
        dictionary = {tree.tag: {k: v[0] if len(
            v) == 1 else v for k, v in dd.items()}}
    if tree.attrib:
        dictionary[tree.tag].update(('@' + k, v)
                                    for k, v in tree.attrib.items())
    if tree.text:
        text = tree.text.strip()
        if children or tree.attrib:
            if text:
                dictionary[tree.tag]['#text'] = text
        else:
            dictionary[tree.tag] = text
    return dictionary


def ReadImage(
        annotation_path: str,
        images_folder_path: str,
        classes: List[str],
        required_image_size: Union[Tuple[int, int], None] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    data = None
    try:
        data = XMLtoDict(ElementTree(file=annotation_path).getroot())[
            "annotation"]
    except Exception as e:
        print("Warning: ", e)
        return None

    image = []
    cls = []
    bounding_boxes = []

    filepath = images_folder_path + '/' + data["filename"]
    image_size = (int(data["size"]["width"]), int(data["size"]["height"]))
    image = cv.imread(filepath)
    if required_image_size:
        image = cv.resize(image, required_image_size)

    annotations = data["object"]
    if isinstance(annotations, dict):
        try:
            cls = [classes.index(annotations["name"])]
        except Exception as e:
            print("Warning: ", e)
            return None

        bounding_boxes = [[
            float(annotations["bndbox"]["xmin"]) / float(image_size[0]),
            float(annotations["bndbox"]["ymin"]) / float(image_size[1]),
            float(annotations["bndbox"]["xmax"]) / float(image_size[0]),
            float(annotations["bndbox"]["ymax"]) / float(image_size[1]),
        ]]
    elif isinstance(annotations, list):
        for annotation in annotations:
            try:
                cls.append(classes.index(annotation["name"]))
            except Exception as e:
                print("Warning: ", e)
                continue
            bounding_boxes.append([
                float(annotation["bndbox"]["xmin"]) / float(image_size[0]),
                float(annotation["bndbox"]["ymin"]) / float(image_size[1]),
                float(annotation["bndbox"]["xmax"]) / float(image_size[0]),
                float(annotation["bndbox"]["ymax"]) / float(image_size[1]),
            ])

    label = yolo.PreprocessLabel(
        cls, num_classes=len(classes), bounding_boxes=bounding_boxes)

    return image, label


def ReadBatched(
        annotation_paths: List[str],
        images_folder_path: str,
        classes: List[str],
        required_image_size: Union[Tuple[int, int], None] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    batch_x = []
    batch_y = []
    for annotation_path in annotation_paths:
        data = ReadImage(
            annotation_path,
            images_folder_path,
            classes,
            required_image_size,
        )
        if data:
            batch_x.append(data[0])
            batch_y.append(data[1])
    return batch_x, batch_y


def BatchedGenerator(
    batch_size: int = 64,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    annotation_dir = "Dataset/VOC2012/Annotations/"
    annotation_paths = [annotation_dir +
                        path for path in listdir(annotation_dir)]
    shuffle(annotation_paths)
    images_path = "Dataset/VOC2012/JPEGImages"
    image_size = (448, 448)
    for i in range(0, len(annotation_paths) - batch_size, batch_size):
        batch_x, batch_y = ReadBatched(annotation_paths[i:i + batch_size], images_path, PASCAL_VOC_CLASSES,
                                       required_image_size=image_size)
        yield np.array(batch_x), np.array(batch_y)
    return None


def ReadData() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    annotation_dir = "Dataset/VOC2012/Annotations/"
    annotation_paths = shuffle([annotation_dir +
                                path for path in listdir(annotation_dir)])
    images_path = "Dataset/VOC2012/JPEGImages"
    image_size = (448, 448)

    return ReadBatched(annotation_paths, images_path, PASCAL_VOC_CLASSES,
                       required_image_size=image_size)
