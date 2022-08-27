# Author: Drew Byron.
# Date: 7/23/22.
# Description of Module:

# Deep learning imports.
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import pytorch_lightning as pl

import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Standard imports.
from pathlib import Path
from typing import List, Union
import gc
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Necessary for creating our images.
from skimage.draw import line_aa


def show(imgs, figsize=(10.0, 10.0)):
    """Displays a single image or list of images. Taken more or less from
    the pytorch docs:
    https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html#visualizing-a-grid-of-images

    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        figsize (Tuple[float, float]): size of figure to display.

    Returns:
        None
    """

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

    return None


def add_labels(img, label, class_map, pred=False, object_count=True):
    """Takes a single image of shape (3, H, W) and adds labels directly
    onto the image using cv2. Used with ImageSegmentation_DS/DM but can
    be used in other applicable computer vision tasks.

    Args:
        img (torch.UInt8Tensor[3, H, W]): a pytorch image.
        label (torch.int64[ds_size, num_classes]): label contians
            either the number of instances of each class (if object_count
            = True) or a binary value representing if
            any of the class are present in the image. For example
            if the image contains 3 instances of class 2 then
            label[1] = 3 if object_count = True and
            label[1] = 1 if object_count = False. Note that here 0 is
            not a valid class so if your class_map contains keys
            0,1,2,3,4 then num_classes = 4.
        class_map (Dict[Dict]): the class map must contain keys that
            correspond to the labels provided. Inner Dict must contain
            "name" and "target_color". class 0 is reserved for the case
            where the image contains no objects (label.sum() == 0).
            A valid example:
            class_map={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "rectangle", "target_color": (255, 0, 0)},
            2: {"name": "line", "target_color": (0, 255, 0)},
            3: {"name": "donut", "target_color": (0, 0, 255)}}.
        pred (bool): whether or not the label provided is a prediction.
            Predictions are printed in the bottom right of the image
            whereas targets are printed in the top left.
        object_count (bool): whether or not the label contains the
            object instance counts or not. See above under label for an
            example.


    Returns:
        img (torch.UInt8Tensor[3, H, W]): a pytorch image with the names
            and (optionally) counts corresponding to the provided label
            drawn over the image.
    """
    img_size = img.shape[-1]
    img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()

    if label.sum() == 0:

        nonzero_classes = [0]
        label_colors = [class_map[0]["target_color"]]
        img_labels = ["background"]

    else:

        nonzero_classes = label.cpu().numpy().nonzero()[0] + 1
        label_colors = [class_map[indx]["target_color"] for indx in nonzero_classes]

        if object_count:
            img_labels = [
                class_map[indx]["name"] + ": {}".format(label[indx - 1])
                for indx in nonzero_classes
            ]
        else:
            img_labels = [class_map[indx]["name"] for indx in nonzero_classes]

    scaling_ratio = img_size / 512

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1 * scaling_ratio
    thickness = 1
    lineType = 1

    y0, x0, dy = 27 * scaling_ratio, 10 * scaling_ratio, 27 * scaling_ratio
    if pred:
        y0, x0 = 400 * scaling_ratio, 315 * scaling_ratio
        thickness = 2

    for i, (img_label, label_color, c) in enumerate(
        zip(img_labels, label_colors, nonzero_classes)
    ):
        y = y0 + c * dy
        fontColor = label_color

        cv2.putText(
            img,
            img_label,
            (int(x0), int(y)),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

    img = torch.from_numpy(img).permute(2, 0, 1)

    return img


def labels_to_masks(labels):
    """Converts  a batch of segmentation labels into binary masks. Used
    with UNET or in other image segmentation tasks. This function works
    for both batches of labels or single (2d) image labels. The Args and
    return descriptions assume a full batch is input.

    Args:
        labels (torch.int64[batch_size, H, W]): a batch of segmentation
            labels. Each pixel is assigned a class (an integer value).

    Returns:
    binary_masks (torch.bool[batch_size, num_obj_ids, H, W]): a batch of
        corresponding binary masks. Layer i (of dim = 1) corresponds to
        a binary mask for class i. The total number of binary masks will
        be the number of unique object ids (num_obj_ids).
    """

    obj_ids = labels.unique()
    if labels.dim() == 2:
        masks = labels == obj_ids[:, None, None]

    if labels.dim() == 3:
        masks = (labels == obj_ids[:, None, None, None]).permute(1, 0, 2, 3)

    return masks


def display_masks_unet(imgs, masks, class_map, alpha=0.4):
    """Takes a batch of images and a batch of masks of the same length and
    overlays the images with the masks using the "target_color" specified
    in the class_map.

    Args:
        imgs (List[torch.ByteTensor[batch_size, 3, H, W]]): a batch of
            images of shape (batch_size, 3, H, W).
        masks (torch.bool[batch_size, num_masks, H, W]]): a batch of
            corresponding boolean masks.
        class_map (Dict[Dict]): the class map must contain keys that
            correspond to the labels provided. Inner Dict must contain
            key "target_color". class 0 is reserved for background.
            A valid example ("name" not necessary):
            class_map={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "rectangle", "target_color": (255, 0, 0)},
            2: {"name": "line", "target_color": (0, 255, 0)},
            3: {"name": "donut", "target_color": (0, 0, 255)}}.
        alpha (float): transparnecy of masks. In range (0-1).

    Returns:
        result_imgs (List[torch.ByteTensor[3, H, W]]]): list of images
            with overlaid segmentation masks.
    """
    num_imgs = len(imgs)

    result_imgs = [
        draw_segmentation_masks(
            imgs[i],
            masks[i],
            alpha=0.4,
            colors=[class_map[j]["target_color"] for j in list(class_map.keys())],
        )
        for i in range(num_imgs)
    ]

    return result_imgs


def display_boxes(imgs, target_pred_dict, class_map, width = 1, fill=False):
    """
    Takes a list of images and a list of target or prediction dictionaries
    of the same len and overlays bounding boxes onto the images.

    Args:
        imgs (List[torch.ByteTensor[3, H, W]]): list of images (each a
            torch.ByteTensor of shape(3, H, W)).

        target_pred_dict (List[Dict[torch.Tensor]]): predictions or targets
            formatted according to the torchvision implimentation of
            FasterRCNN and MaskRCNN.
            See link below for details on the target/prediction formatting.
            https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

        class_map (Dict[Dict]): the class map must contain keys that
            correspond to the labels provided. Inner Dict must contain
            key "target_color". class 0 is reserved for background.
            A valid example ("name" not necessary):
            class_map={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "rectangle", "target_color": (255, 0, 0)},
            2: {"name": "line", "target_color": (0, 255, 0)},
            3: {"name": "donut", "target_color": (0, 0, 255)}}.

        fill (bool): if True the inside of the bounding boxes will be
            filled with color.

    Returns:
        result_imgs (List[torch.ByteTensor[3, H, W]]): list of images with
            overlaid bounding boxes.
    """
    num_imgs = len(imgs)
    result_imgs = [
        draw_bounding_boxes(
            imgs[i],
            target_pred_dict[i]["boxes"].int(),
            fill=fill,
            colors=[
                class_map[j.item()]["target_color"]
                for j in target_pred_dict[i]["labels"]
            ],
            width=width,
        )
        for i in range(num_imgs)
    ]

    return result_imgs


def display_masks_rcnn(imgs, target_pred_dict, class_map, threshold=0.5, alpha=0.4):
    """
    Takes a list of images and a list of target or prediction dictionaries
    of the same len and overlays segmentation masks onto the images.

    Args:
        imgs (List[torch.ByteTensor[3, H, W]]): list of images (each a
            torch.ByteTensor of shape(3, H, W)).

        target_pred_dict (List[Dict[torch.Tensor]]): predictions or targets
            formatted according to the torchvision implimentation of
            FasterRCNN and MaskRCNN.
            See link below for details on the target/prediction formatting.
            https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

        class_map (Dict[Dict]): the class map must contain keys that
            correspond to the labels provided. Inner Dict must contain
            key "target_color". class 0 is reserved for background.
            A valid example ("name" not necessary):
            class_map={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "rectangle", "target_color": (255, 0, 0)},
            2: {"name": "line", "target_color": (0, 255, 0)},
            3: {"name": "donut", "target_color": (0, 0, 255)}}.

        threshold (float): threshold applied to soft masks. In range (0-1).

        alpha (float): transparnecy of masks. In range (0-1).

    Returns:
        result_imgs (List[torch.ByteTensor[3, H, W]]): list of images with
            overlaid segmentation masks.
    """
    num_imgs = len(imgs)

    if target_pred_dict[0]["masks"].dtype == torch.float32:
        target_pred_dict = threshold_pred_masks(target_pred_dict, threshold)

    result_imgs = [
        draw_segmentation_masks(
            imgs[i],
            target_pred_dict[i]["masks"],
            alpha=alpha,
            colors=[
                class_map[j.item()]["target_color"]
                for j in target_pred_dict[i]["labels"]
            ],
        )
        for i in range(num_imgs)
    ]

    return result_imgs


def display_labels(imgs, target_pred_dict, class_map, text_size, text_width):
    """
    Takes a list of images and a list of target or prediction dictionaries
    of the same len and adds labels to the instances. Note that for very
    small images this will behave poorly.

    Args:
        imgs (List[torch.ByteTensor[3, H, W]]): list of images (each a
            torch.ByteTensor of shape(3, H, W)).

        target_pred_dict (List[Dict[torch.Tensor]]): predictions or targets
            formatted according to the torchvision implimentation of
            FasterRCNN and MaskRCNN.
            See link below for details on the target/prediction formatting.
            https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

        class_map (Dict[Dict]): the class map must contain keys that
            correspond to the labels provided. Inner Dict must contain
            key "target_color". class 0 is reserved for background.
            class_map={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "rectangle", "target_color": (255, 0, 0)},
            2: {"name": "line", "target_color": (0, 255, 0)},
            3: {"name": "donut", "target_color": (0, 0, 255)}}.

        text size (int): size of instance label text.

        text_width (int): width of instance label text.

    Returns:
        labeled_imgs (List[torch.ByteTensor[3, H, W]]): list of images with
            overlaid instance labels.
    """

    labeled_imgs = []
    for i, img in enumerate(imgs):

        ndarr_img = img.permute(1, 2, 0).cpu().numpy().copy()

        class_names = [
            class_map[j.item()]["name"] for j in target_pred_dict[i]["labels"]
        ]
        class_colors = [
            class_map[j.item()]["target_color"] for j in target_pred_dict[i]["labels"]
        ]
        box_locations = [
            (int(box[0]), int(box[1])) for box in target_pred_dict[i]["boxes"]
        ]

        for name, color, box_loc in zip(class_names, class_colors, box_locations):

            ndarr_img = cv2.putText(
                ndarr_img,
                name,
                box_loc,
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                (0, 0, 0),
                thickness=text_width,
            )

        labeled_imgs.append(
            torch.from_numpy(ndarr_img).permute(2, 0, 1).to(dtype=torch.uint8)
        )

    return labeled_imgs


def threshold_pred_masks(preds, threshold=0.5):
    """
    Takes a list of prediction dictionaries (one for each image) and
    thresholds the soft masks, returning a list of prediction dictionaries
    with thresholded (boolean) masks.

    Args:
        preds (List[Dict[torch.Tensor]]): predictions as output by the
            torchvision implimentation of MaskRCNN. The masks consist of
            probabilities (torch.float32) in the range (0,1) for each pixel.
            See link below for details on the target/prediction formatting.
            https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

    Returns:
        thresholded_preds (List[Dict[torch.Tensor]]): predictions with
            boolean (torch.bool) masks.
    """

    thresholded_preds = [
        {**pred, "masks": (pred["masks"] > threshold).squeeze(dim=1)} for pred in preds
    ]

    return thresholded_preds


def build_coco_class_map(seed, drop_background=True):
    """
    Returns a class_map for coco classes.

    Args:
        seed (int): seed to use to build np rng. Can change this to get
            a new set of colors.
        drop_background (bool): If true the background class (assigned
            class_id = 0 by default) will be dropped from the class map.
            This is the default because in displaying the segmented images
            often one doesn't care to display background.

    Returns:
        coco_class_map (Dict[Dict]): class_map to be used with other functions
            in this module.
    """

    COCO_CLASS_NAMES = [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    rng = np.random.default_rng(seed)
    COLORS = np.around(rng.uniform(100, 255, size=(len(COCO_CLASS_NAMES), 3))).astype(
        int
    )
    # By making some of the values zero you make the colors less brown.
    indices = rng.choice(
        np.arange(COLORS.size), replace=False, size=int(COLORS.size * 0.33)
    )

    COLORS[np.unravel_index(indices, COLORS.shape)] = 0

    coco_class_map = {}
    for i, name in enumerate(COCO_CLASS_NAMES):
        if i == 0 and drop_background:
            continue
        coco_class_map[i] = {"name": name, "target_color": tuple(COLORS[i])}

    return coco_class_map


def apply_score_cut(preds, score_threshold=0.5):
    """
    Takes a list of prediction dictionaries (one for each image) and cuts
    out all instances whose score is below the score threshold.

    Args:
        preds (List[Dict[torch.Tensor]]): predictions as output by the
            torchvision implimentation of MaskRCNN or FasterRCNN. The 
            scores are in the range (0,1) and signify the certainty of 
            the model for that instance.
            See link below for details on the target/prediction formatting.
            https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html
        score_threshold (float): the threshold to apply to the identified
            objects. If an instance is below the score_threshold it will
            be removed from the score_thresholded_preds dictionary.

    Returns:
        score_thresholded_preds (List[Dict[torch.Tensor]]): predictions
            that exceed score_threshold.
    """
    score_thresholded_preds = [
        {key: value[pred["scores"] > score_threshold] for key, value in pred.items()}
        for pred in preds
    ]

    return score_thresholded_preds


def load_img_dir(path, resize_factor=0.5):
    """
    Loads all of the images in a directory into torch images.

    Args:
        path (str): path should point to a directory that only contains
            .JPG images. Or any image type compatible with cv2.imread().

        resize_factor (float): how to resize the image. Often one would
            like to reduce the size of the images to be easier/faster to
            use with our maskrcnn model.

    Returns:
        imgs (List[torch.ByteTensor[3, H, W]]): list of images (each a
            torch.ByteTensor of shape(3, H, W)).
    """
    path_glob = Path(path).glob("**/*")
    files = [x for x in path_glob if x.is_file()]
    if len(files) == 0:
        raise UserWarning("No files found at the input path.")

    imgs = []
    for file in files:

        img = cv2.imread(str(file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)
        img_resized = TF.resize(img, size=[int(img.shape[1] * resize_factor)])
        imgs.append(img_resized)

    return imgs


def get_preds(imgs, maskrcnn, device):
    """
    Simple utility function for returning the predictions of maskrcnn model.
    This deals with putting the model and imgs on device and normalizing
    torch.ByteTensor.

    Args:
        maskrcnn (nn.Module): an instance of the torchvision Mask RCNN
            model. One can build with following call: maskrcnn =
            cv_models.get_maskrcnn(num_classes=-1, pretrained=True)

        device (str): what device to put model and imgs. Use following
            call: device = "cuda" if torch.cuda.is_available() else "cpu"

    Returns:
        preds (List[Dict[torch.Tensor]]): predictions as output by the
            torchvision implimentation of MaskRCNN.
            See link below for details on the target/prediction formatting.
            https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html
    """

    maskrcnn = maskrcnn.to(device)
    maskrcnn.eval()
    imgs_normed = [(img / 255.0).to(device) for img in imgs]

    preds = maskrcnn(imgs_normed)

    return preds


def save_imgs(imgs, base_path, base_name):
    """
    Saves torch images to JPG file format.

    Args:
        imgs (List[torch.ByteTensor[3, H, W]]): list of images (each a
            torch.ByteTensor of shape(3, H, W)).

        base_path (str): path to directory where images should be written.

        base_name (str): base name to be used to build JPG file paths.

    Returns:
        None
    """
    base_path = Path(base_path)
    for i, img in enumerate(imgs):

        file_path = base_path / Path(base_name + f"_{i}" + ".JPG")

        ndarr_img = img.permute(1, 2, 0).cpu().numpy()
        ndarr_img = cv2.cvtColor(ndarr_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(file_path), ndarr_img)

    return None


def maskrcnn_process_images(imgs, preds, class_map, config):
    """
    Processes a set of imgs and associated predictions.

    Args:
        imgs (List[torch.ByteTensor[3, H, W]]): list of images (each a
            torch.ByteTensor of shape(3, H, W)).

        preds (List[Dict[torch.Tensor]]): predictions as output by the
            torchvision implimentation of MaskRCNN.
            See link below for details on the target/prediction formatting.
            https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

        class_map (Dict[Dict]): the class map must contain keys that
            correspond to the labels provided. Inner Dict must contain
            key "target_color". class 0 is reserved for background.
            class_map={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "rectangle", "target_color": (255, 0, 0)},
            2: {"name": "line", "target_color": (0, 255, 0)},
            3: {"name": "donut", "target_color": (0, 0, 255)}}.

        config (Dict): a config dictionary that contains info on how to
            process the imgs. Example below. Must contain all keys included
            in the example:
            config = {"boxes": True,
                      "masks": True,
                      "labels": True,
                      "score_cut": .5,
                      "box_width": 2,
                      "box_fill" : False,
                      "mask_threshold": .5,
                      "mask_alpha": .5,
                      "label_size": 1,
                      "label_width": 1}

    Returns:
        processed_imgs (List[torch.ByteTensor[3, H, W]]): list of processed
            images (each a torch.ByteTensor of shape(3, H, W)).
    """

    if len(config) != 10:
        raise UserWarning(
            "config dictionary must conain all 10 elements included in the documentation."
        )

    preds_cut = apply_score_cut(preds, config["score_cut"])

    processed_imgs = imgs
    if config["boxes"]:
        processed_imgs = display_boxes(
            processed_imgs,
            preds_cut,
            class_map,
            width=config["box_width"],
            fill=config["box_fill"],
        )
    if config["masks"]:
        processed_imgs = display_masks_rcnn(
            processed_imgs,
            preds_cut,
            class_map,
            threshold=config["mask_threshold"],
            alpha=config["mask_alpha"],
        )
    if config["labels"]:
        processed_imgs = display_labels(
            processed_imgs,
            preds_cut,
            class_map,
            text_size=config["label_size"],
            text_width=config["label_width"],
        )

    return processed_imgs


def maskrcnn_process_video(
    raw_video_path,
    processed_video_path,
    device,
    maskrcnn,
    class_map,
    config,
    output_shape=(700, 700),
    show_first_frame=False,
    frame_max=1e4,
    fps=30,
):
    """
    Processes a .MOV, adding segmentation, labels, and/or bboxes and
    writes out a processed version of the original .MOV

    Args:

        raw_video_path (str): path to raw .MOV file. Should work with any
            file type compatible with cv2.VideoCapture.

        processed_video_path (str): path where processed video will be
            written.

        device (str): what device to put model and imgs. Use following
            call: device = "cuda" if torch.cuda.is_available() else "cpu"

        maskrcnn (nn.Module): an instance of the torchvision Mask RCNN
            model. One can build with following call: maskrcnn =
            cv_models.get_maskrcnn(num_classes=-1, pretrained=True)

        class_map (Dict[Dict]): the class map must contain keys that
            correspond to the labels provided. Inner Dict must contain
            key "target_color". class 0 is reserved for background.
            class_map={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "rectangle", "target_color": (255, 0, 0)},
            2: {"name": "line", "target_color": (0, 255, 0)},
            3: {"name": "donut", "target_color": (0, 0, 255)}}.

        config (Dict): a config dictionary that contains info on how to
            process the imgs. Example below. Must contain all keys included
            in the example:
            config = {"boxes": True,
                      "masks": True,
                      "labels": True,
                      "score_cut": .5,
                      "box_width": 2,
                      "box_fill" : False,
                      "mask_threshold": .5,
                      "mask_alpha": .5,
                      "label_size": 1,
                      "label_width": 1}

        output_shape (tuple[int,int]): size of the ouput video.
        show_first_frame (bool): used to sanity check. Set to true to see the
            result of first frame.
        frame_max (int): used to limit the time this takes and to sanity
            check things. Set to 10 to be sure things are working.
        fps (int): frames per second of video. Default for google photos
            is 30.

    Returns:
        None
    """
    if not Path(raw_video_path).is_file():
        raise UserWarning("No file found at raw_video_path.")

    # Capture for reading raw video.
    cap = cv2.VideoCapture(str(raw_video_path))

    # Writer for writing to processed video.
    out = cv2.VideoWriter(
        str(processed_video_path),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        output_shape,
    )

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    frame_count = 0
    # Read until video is completed
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:

            # resize frame to match output_shape
            resized_frame = cv2.resize(frame, output_shape)
            img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1)

            imgs = [img]
            preds = get_preds(imgs, maskrcnn, device)

            # Process the frames one at a time.
            processed_imgs = maskrcnn_process_images(imgs, preds, class_map, config)

            if frame_count == 0 and show_first_frame:
                show(processed_imgs)

            ndarr_img = processed_imgs[0].permute(1, 2, 0).cpu().numpy()
            ndarr_img = cv2.cvtColor(ndarr_img, cv2.COLOR_BGR2RGB)
            out.write(ndarr_img)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames.")
            if frame_count > frame_max:
                break

        else:
            break

    print(f"Processed {frame_count} total frames.")

    # Release the video writer object.
    out.release()

    # Release the video capture object.
    cap.release()

    return None


