# Author: Drew Byron.
# Date: 7/23/22.
# Description of Module:

# Standard imports.
import numpy as np
from skimage.draw import line_aa

# Deep learning imports.
import torch
from torch import nn

import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader, random_split, TensorDataset

from torchmetrics.detection.mean_ap import MeanAveragePrecision


# TODOs:
# * Recomment the class_map.
# * Comment out Draw.
# * Reformat to match new base class. Get this working before commenting everything.
# * Test it all out in the different nbs.


class Draw:
    """
    Class used to draw shapes onto images. Methods return coordinates of
    corresponding shape on a 2d np array of shape (img_size, img_size).
    The np rng is used for enabling derministic behaviour.

    Args:
    img_size (int): draws onto 2d array of shape (img_size, img_size).
    rng (Generator): used for enabling deterministic behaviour. Example
        of valid rng: rng = np.random.default_rng(12345)

    """

    def __init__(self, img_size, rng):

        self.img_size = img_size
        self.rng = rng

        return None

    def rectangle(self):
        """
        Returns the image coordinates of a rectangle.
        """
        # Min and max rectangle height.
        a = self.rng.integers(self.img_size * 3 / 20, self.img_size * 7 / 20)
        b = self.rng.integers(self.img_size * 3 / 20, self.img_size * 7 / 20)

        # Initial coordinates of the rectangle.
        xx, yy = np.where(np.ones((a, b)) == 1)

        # Place the rectangle randomly in the image.
        cx = self.rng.integers(0 + a, self.img_size - a)
        cy = self.rng.integers(0 + b, self.img_size - b)

        rectangle = xx + cx, yy + cy

        return rectangle

    def line(self):
        """
        Returns the image coordinates of a line.
        """
        # Randomly choose the start and end coordinates.
        a, b = self.rng.integers(0, self.img_size / 3, size=2)
        c, d = self.rng.integers(self.img_size / 2, self.img_size, size=2)

        # Flip a coin to see if slope of line is + or -.
        coin_flip = self.rng.integers(low=0, high=2)
        # Use a skimage.draw method to draw the line.
        if coin_flip:
            xx, yy, _ = line_aa(a, b, c, d)
        else:
            xx, yy, _ = line_aa(a, d, c, b)

        line = xx, yy

        return line

    def donut(self):
        """
        Returns the image coordinates of an elliptical donut.
        """
        # Define a grid
        xx, yy = np.mgrid[: self.img_size, : self.img_size]
        cx = self.rng.integers(0, self.img_size)
        cy = self.rng.integers(0, self.img_size)

        # Define the width of the donut.
        width = self.rng.uniform(self.img_size / 3, self.img_size)

        # Give the donut some elliptical nature.
        e0 = self.rng.uniform(0.1, 5)
        e1 = self.rng.uniform(0.1, 5)

        # Use the forumula for an ellipse.
        ellipse = e0 * (xx - cx) ** 2 + e1 * (yy - cy) ** 2

        donut = (ellipse < (self.img_size + width)) & (
            ellipse > (self.img_size - width)
        )

        return donut


class CV_DS_Base(torch.utils.data.Dataset):
    """
    Base class for a set of PyTorch computer vision datasets. This class
    contains all of the attributes and methods common to all datasets
    in this package.
    Alone this base class has no functionality. The utility of these datasets
    is that they enable the user to test cv models with very small and
    simple images with tunable complexity. It also requires no downloading
    of images and one can scale the size of the datasets easily.

    Args:
        ds_size (int): number of images in dataset.
        img_size (int): will build images of shape (3, img_size, img_size).
        shapes_per_image (Tuple[int, int]): will produce images containing
            minimum number of shapes Tuple[0] and maximum number of shapes
            Tuple[1]. For example shapes_per_image = (2,2) would create a
            dataset where each image contains exactly two shapes.
        class_probs (Tuple[float, float, float]): relative probability of
            each shape occuring in an image. Need not sum to 1. For example
            class_probs = (1,1,0) will create a dataset with 50% class 1
            shapes, 50% class 2 shapes, 0% class 3 shapes.
        rand_seed (int): used to instantiate a numpy random number generator.
        class_map (Dict[Dict]): the class map must contain keys (0,1,2,3)
            and contain names "background", "rectangle", "line", and "donut".
            "gs_range" specifies the upper and lower bound of the
            grayscale values (0, 255) used to color the shapes.
            "target_color" can be used by visualization tools to assign
            a color to masks and boxes. Note that class 0 is reserved for
            background in most instance seg models, so one can rearrange
            the class assignments of different shapes but 0 must correspond
            to "background". The utility of this Dict is to enable the user
            to change target colors, class assignments, and shape
            intensities. A valid example:
            class_map={
            0: {"name": "background","gs_range": (200, 255),"target_color": (255, 255, 255),},
            1: {"name": "rectangle", "gs_range": (0, 100), "target_color": (255, 0, 0)},
            2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
            3: {"name": "donut", "gs_range": (0, 100), "target_color": (0, 0, 255)}}.
    """

    def __init__(
        self,
        ds_size,
        img_size,
        shapes_per_image,
        class_probs,
        rand_seed,
        class_map,
    ):

        if img_size <= 20:
            raise ValueError(
                "Different shapes are hard to distinguish for images of shape (3, 20, 20) or smaller."
            )

        if sorted(list(class_map.keys())) != sorted([0, 1, 2, 3]):
            raise ValueError("Dict class_map must contain keys 0,1,2,3.")

        self.ds_size = ds_size
        self.img_size = img_size
        self.rand_seed = rand_seed
        self.shapes_per_image = shapes_per_image
        self.class_probs = np.array(class_probs) / np.array(class_probs).sum()
        self.class_map = class_map

        self.rng = np.random.default_rng(self.rand_seed)

        self.draw = Draw(self.img_size, self.rng)

        self.class_ids = np.array([1, 2, 3])
        self.num_shapes_per_img = self.rng.integers(
            low=self.shapes_per_image[0],
            high=self.shapes_per_image[1] + 1,
            size=self.ds_size,
        )

        self.chosen_ids_per_img = [
            self.rng.choice(a=self.class_ids, size=num_shapes, p=self.class_probs)
            for num_shapes in self.num_shapes_per_img
        ]

        self.imgs, self.targets = [], []

        return None

    def __getitem__(self, idx):

        return self.imgs[idx], self.targets[idx]

    def __len__(self):

        return len(self.imgs)

    def draw_shape(self, class_id):
        """
        Draws the shape with the associated class_id.
        """

        if self.class_map[class_id]["name"] == "rectangle":
            shape = self.draw.rectangle()
        elif self.class_map[class_id]["name"] == "line":
            shape = self.draw.line()
        elif self.class_map[class_id]["name"] == "donut":
            shape = self.draw.donut()

        else:
            raise ValueError(
                "You must include rectangle, donut, and line in your class_map."
            )

        return shape


class ObjectCounting_DS(CV_DS_Base):
    """
    Self contained PyTorch Dataset for testing object counting models.

    Args:
        ds_size (int): number of images in dataset.
        img_size (int): will build images of shape (3, img_size, img_size).
        shapes_per_image (Tuple[int, int]): will produce images containing
            minimum number of shapes Tuple[0] and maximum number of shapes
            Tuple[1]. For example shapes_per_image = (2,2) would create a
            dataset where each image contains exactly two shapes.
        class_probs (Tuple[float, float, float]): relative probability of
            each shape occuring in an image. Need not sum to 1. For example
            class_probs = (1,1,0) will create a dataset with 50% class 1
            shapes, 50% class 2 shapes, 0% class 3 shapes.
        rand_seed (int): used to instantiate a numpy random number generator.
        class_map (Dict[Dict]): the class map must contain keys (0,1,2,3)
            and contain names "background", "rectangle", "line", and "donut".
            "gs_range" specifies the upper and lower bound of the
            grayscale values (0, 255) used to color the shapes.
            "target_color" can be used by visualization tools to assign
            a color to masks and boxes. Note that class 0 is reserved for
            background in most instance seg models, so one can rearrange
            the class assignments of different shapes but 0 must correspond
            to "background". The utility of this Dict is to enable the user
            to change target colors, class assignments, and shape
            intensities. A valid example:
            class_map={
            0: {"name": "background","gs_range": (200, 255),"target_color": (255, 255, 255),},
            1: {"name": "rectangle", "gs_range": (0, 100), "target_color": (255, 0, 0)},
            2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
            3: {"name": "donut", "gs_range": (0, 100), "target_color": (0, 0, 255)}}.
        object_count (bool): whether or not the targets contain the
            object instance counts or not. Example below under the
            build_imgs_and_targets() method.
    """

    def __init__(
        self,
        ds_size=100,
        img_size=256,
        shapes_per_image=(1, 3),
        class_probs=(1, 1, 1),
        rand_seed=12345,
        class_map={
            0: {
                "name": "background",
                "gs_range": (200, 255),
                "target_color": (255, 255, 255),
            },
            1: {"name": "rectangle", "gs_range": (0, 100), "target_color": (255, 0, 0)},
            2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
            3: {"name": "donut", "gs_range": (0, 100), "target_color": (0, 0, 255)},
        },
        object_count=False,
    ):

        super().__init__(
            ds_size, img_size, shapes_per_image, class_probs, rand_seed, class_map
        )

        self.object_count = object_count
        self.num_classes = np.max(np.nonzero(self.class_probs)) + 1

        self.imgs, self.targets = self.build_imgs_and_targets()

    def build_imgs_and_targets(self):
        """
        Builds images and targets for object counting.

        Returns:
            imgs (torch.UInt8Tensor[ds_size, 3, img_size, img_size]): images
                containing different shapes. The images are gray-scale
                (each layer of the first (color) dimension is identical).
                This makes it easier to visualize targets and predictions.
            targets (torch.int64[ds_size, num_classes]): targets contian
                either the number of instances of each class (if
                object_count = True) or a binary value representing if
                any of the class are present in the image. For example
                if image i contains 3 instances of class 2 then
                targets[i][1] = 3 if object_count = True and
                targets[i][1] = 1 if object_count = False.
        """
        imgs = []
        targets = []

        for idx in range(self.ds_size):

            chosen_ids = self.chosen_ids_per_img[idx]

            # Creating an empty noisy img.
            img = self.rng.integers(
                self.class_map[0]["gs_range"][0],
                self.class_map[0]["gs_range"][1],
                (self.img_size, self.img_size),
            )

            target = np.zeros(self.num_classes)

            # Filling the noisy img with shapes and building targets.
            for i, class_id in enumerate(chosen_ids):
                shape = self.draw_shape(class_id)
                gs_range = self.class_map[class_id]["gs_range"]
                img[shape] = self.rng.integers(
                    gs_range[0], gs_range[1], img[shape].shape
                )
                target[class_id - 1] += 1

            # Convert from np to torch and assign appropriate dtypes.
            img = torch.from_numpy(img)
            img = img.unsqueeze(dim=0).repeat(3, 1, 1).type(torch.ByteTensor)

            target = torch.from_numpy(target).long()
            if not self.object_count:
                target = torch.clamp(target, max=1)
            imgs.append(img)
            targets.append(target)

        # Turn a list of imgs with shape (3, H, W) of len ds_size to a tensor
        # of shape (ds_size, 3, H, W)
        imgs = torch.stack(imgs)
        targets = torch.stack(targets)

        return imgs, targets


class ImageSegmentation_DS(CV_DS_Base):
    """
    Self contained PyTorch Dataset for testing image segmentation models.

    Args:
        ds_size (int): number of images in dataset.
        img_size (int): will build images of shape (3, img_size, img_size).
        shapes_per_image (Tuple[int, int]): will produce images containing
            minimum number of shapes Tuple[0] and maximum number of shapes
            Tuple[1]. For example shapes_per_image = (2,2) would create a
            dataset where each image contains exactly two shapes.
        class_probs (Tuple[float, float, float]): relative probability of
            each shape occuring in an image. Need not sum to 1. For example
            class_probs = (1,1,0) will create a dataset with 50% class 1
            shapes, 50% class 2 shapes, 0% class 3 shapes.
        rand_seed (int): used to instantiate a numpy random number generator.
        class_map (Dict[Dict]): the class map must contain keys (0,1,2,3)
            and contain names "background", "rectangle", "line", and "donut".
            "gs_range" specifies the upper and lower bound of the
            grayscale values (0, 255) used to color the shapes.
            "target_color" can be used by visualization tools to assign
            a color to masks and boxes. Note that class 0 is reserved for
            background in most instance seg models, so one can rearrange
            the class assignments of different shapes but 0 must correspond
            to "background". The utility of this Dict is to enable the user
            to change target colors, class assignments, and shape
            intensities. A valid example:
            class_map={
            0: {"name": "background","gs_range": (200, 255),"target_color": (255, 255, 255),},
            1: {"name": "rectangle", "gs_range": (0, 100), "target_color": (255, 0, 0)},
            2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
            3: {"name": "donut", "gs_range": (0, 100), "target_color": (0, 0, 255)}}.
    """

    def __init__(
        self,
        ds_size=100,
        img_size=256,
        shapes_per_image=(1, 3),
        class_probs=(1, 1, 1),
        rand_seed=12345,
        class_map={
            0: {
                "name": "background",
                "gs_range": (200, 255),
                "target_color": (255, 255, 255),
            },
            1: {"name": "rectangle", "gs_range": (0, 100), "target_color": (255, 0, 0)},
            2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
            3: {"name": "donut", "gs_range": (0, 100), "target_color": (0, 0, 255)},
        },
        object_count=False,
    ):

        super().__init__(
            ds_size, img_size, shapes_per_image, class_probs, rand_seed, class_map
        )

        self.imgs, self.targets = self.build_imgs_and_targets()

    def build_imgs_and_targets(self):

        """
        Builds images and labels.

        Returns:
            imgs (torch.UInt8Tensor[ds_size, 3, img_size, img_size]): images
                containing different shapes. The images are gray-scale
                (each layer of the first (color) dimension is identical).
                This makes it easier to visualize targets and predictions.
            targets (torch.UInt8Tensor[ds_size, img_size, img_size]):
                corresponding segmentation labels. Each pixel is assigned
                a class label, with class_id = 0 reserved for background.

        """
        imgs = []
        targets = []

        for idx in range(self.ds_size):

            chosen_ids = self.chosen_ids_per_img[idx]

            # Creating an empty noisy img.
            img = self.rng.integers(
                self.class_map[0]["gs_range"][0],
                self.class_map[0]["gs_range"][1],
                (self.img_size, self.img_size),
            )

            # Initially all pixels are labeled zero (background).
            target = np.zeros((self.img_size, self.img_size))

            # Filling the noisy img with shapes and building targets.
            for i, class_id in enumerate(chosen_ids):
                shape = self.draw_shape(class_id)
                gs_range = self.class_map[class_id]["gs_range"]
                img[shape] = self.rng.integers(
                    gs_range[0], gs_range[1], img[shape].shape
                )
                target[shape] = class_id

            # Convert from np to torch and assign appropriate dtypes.
            img = torch.from_numpy(img)
            img = img.unsqueeze(dim=0).repeat(3, 1, 1).type(torch.ByteTensor)

            target = torch.from_numpy(target).long()
            imgs.append(img)
            targets.append(target)

        # Turn a list of imgs with shape (3, H, W) of len ds_size to a tensor
        # of shape (ds_size, 3, H, W)
        imgs = torch.stack(imgs)
        targets = torch.stack(targets)

        return imgs, targets


class ObjectDetection_DS(CV_DS_Base):
    """
    Self contained PyTorch Dataset for testing object detection and
    instance segmentation models.
    Note that the specifics of the target formatting is adherent to the
    requirements of the torchvision MaskRCNN and FasterRCNN implimentations.
    That said, this dataset should work with any object detection or
    instance segmentation model that requires the same target formatting
    (such as YOLO).
    See the MaskRCNN documentation (linked below) for more details on the
    formatting of the targets.
    https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

    Args:
        ds_size (int): number of images in dataset.
        img_size (int): will build images of shape (3, img_size, img_size).
        shapes_per_image (Tuple[int, int]): will produce images containing
            minimum number of shapes Tuple[0] and maximum number of shapes
            Tuple[1]. For example shapes_per_image = (2,2) would create a
            dataset where each image contains exactly two shapes.
        class_probs (Tuple[float, float, float]): relative probability of
            each shape occuring in an image. Need not sum to 1. For example
            class_probs = (1,1,0) will create a dataset with 50% class 1
            shapes, 50% class 2 shapes, 0% class 3 shapes.
        rand_seed (int): used to instantiate a numpy random number generator.
        class_map (Dict[Dict]): the class map must contain keys (0,1,2,3)
            and contain names "background", "rectangle", "line", and "donut".
            "gs_range" specifies the upper and lower bound of the
            grayscale values (0, 255) used to color the shapes.
            "target_color" can be used by visualization tools to assign
            a color to masks and boxes. Note that class 0 is reserved for
            background in most instance seg models, so one can rearrange
            the class assignments of different shapes but 0 must correspond
            to "background". The utility of this Dict is to enable the user
            to change target colors, class assignments, and shape
            intensities. A valid example:
            class_map={
            0: {"name": "background","gs_range": (200, 255),"target_color": (255, 255, 255),},
            1: {"name": "rectangle", "gs_range": (0, 100), "target_color": (255, 0, 0)},
            2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
            3: {"name": "donut", "gs_range": (0, 100), "target_color": (0, 0, 255)}}.
        target_masks (bool): whether or not the target dictionaries should
            contain boolean masks for each object instance. Masks are not
            necessary to train FasterRCNN or other object detection models
            but are necessary to train instance segmentation models such
            as MaskRCNN.
    """

    def __init__(
        self,
        ds_size=100,
        img_size=256,
        shapes_per_image=(1, 3),
        class_probs=(1, 1, 1),
        rand_seed=12345,
        class_map={
            0: {
                "name": "background",
                "gs_range": (200, 255),
                "target_color": (255, 255, 255),
            },
            1: {"name": "rectangle", "gs_range": (0, 100), "target_color": (255, 0, 0)},
            2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
            3: {"name": "donut", "gs_range": (0, 100), "target_color": (0, 0, 255)},
        },
        target_masks=False,
    ):

        super().__init__(
            ds_size, img_size, shapes_per_image, class_probs, rand_seed, class_map
        )

        self.target_masks = target_masks
        self.imgs, self.targets = self.build_imgs_and_targets()

    def build_imgs_and_targets(self):
        """
        Builds images and targets for object detection and instance
        segmentation.

        Returns:
            imgs (torch.UInt8Tensor[ds_size, 3, img_size, img_size]): images
                containing different shapes. The images are gray-scale
                (each layer of the first (color) dimension is identical).
                This makes it easier to visualize targets and predictions.
            targets (List[Dict[torch.Tensor]]): list of dictionaries of
                length ds_size. Each image has an associated target
                dictionary that contains the following (note that N = number
                of instances/shapes in a given img):
                - boxes (torch.FloatTensor[N, 4]): the ground-truth boxes
                in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                0 <= y1 < y2 <= H.
                - labels (torch.Int64Tensor[N]): the class label for each
                ground-truth box.
                - masks (torch.UInt8Tensor[N, H, W]): the segmentation
                binary masks for each instance. Not included if
                target_masks = False.

        """
        imgs = []
        targets = []

        for idx in range(self.ds_size):

            chosen_ids = self.chosen_ids_per_img[idx]
            num_shapes = self.num_shapes_per_img[idx]

            # Creating an empty noisy img.
            img = self.rng.integers(
                self.class_map[0]["gs_range"][0],
                self.class_map[0]["gs_range"][1],
                (self.img_size, self.img_size),
            )
            # Empty target dictionary to be filled.
            target = {}
            masks = np.zeros((num_shapes, self.img_size, self.img_size))

            # Filling the noisy img with shapes and building masks.
            for i, class_id in enumerate(chosen_ids):
                shape = self.draw_shape(class_id)
                gs_range = self.class_map[class_id]["gs_range"]
                img[shape] = self.rng.integers(
                    gs_range[0], gs_range[1], img[shape].shape
                )
                masks[i][shape] = 1

            # Convert from np to torch and assign appropriate dtypes.
            img = torch.from_numpy(img)
            img = img.unsqueeze(dim=0).repeat(3, 1, 1).type(torch.ByteTensor)

            masks = torch.from_numpy(masks).bool()
            chosen_ids = torch.from_numpy(chosen_ids).long()

            # Fill in the target dictionary.
            if self.target_masks:
                target["masks"] = masks
            target["boxes"] = masks_to_boxes(masks)
            target["area"] = self.boxes_area(target["boxes"])
            target["labels"] = chosen_ids
            target["image_id"] = torch.tensor([idx])
            target["iscrowd"] = torch.zeros((num_shapes,), dtype=torch.int64)

            targets.append(target)
            imgs.append(img)

        # Turn a list of imgs with shape (3, H, W) of len ds_size to a tensor
        # of shape (ds_size, 3, H, W)
        imgs = torch.stack(imgs)

        return imgs, targets

    def boxes_area(self, boxes):
        """
        Returns the area of a bounding box with [x1, y1, x2, y2] format, with
        0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        """
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return area
