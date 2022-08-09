# Author: Drew Byron.
# Date: 7/23/22.
# Description of Module:

import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid


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


def add_labels(img, label, class_map, pred=False, object_count=False):
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


def display_boxes(imgs, target_pred_dict, class_map, fill=False):
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
            target_pred_dict[i]["boxes"],
            fill=fill,
            colors=[
                class_map[j.item()]["target_color"]
                for j in target_pred_dict[i]["labels"]
            ],
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
            alpha=0.4,
            colors=[
                class_map[j.item()]["target_color"]
                for j in target_pred_dict[i]["labels"]
            ],
        )
        for i in range(num_imgs)
    ]

    return result_imgs


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
