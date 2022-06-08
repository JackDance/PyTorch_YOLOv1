from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
from .cocodataset import coco_class_index, coco_class_labels, COCODataset, coco_root
from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean, std):
    """
    普通的图像归一化操作，即将图像的所有像素都除以255，这样所有像素有映射到了0-1范围内，随后再使用均值和标准差做
    进一步的归一化处理。
    Args:
        image: 已经通过opencv读取出的图像
        size: 要resize的大小
        mean: 均值，默认使用ImageNet的mean
        std: 标准差，默认使用ImageNet的std

    Returns:
            返回归一化的图像x
    """
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x /= 255.
    x -= mean
    x /= std
    return x


class BaseTransform:
    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean, self.std), boxes, labels
