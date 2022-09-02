<p align="center"><img width="15%" src="/images/logos/pytorch_logo.png" /></p>

--------------------------------------------------------------------------------

<p align="center"><img width="50%" src="/images/logos/pytorch_tutorials_logo_cv.png" /></p>

--------------------------------------------------------------------------------

<p align="center"><img width="40%" src="/images/intro_to_cv/mt_rainier_seg_0.JPG" />              <img width="40%" src="/images/intro_to_cv/mt_rainier_seg_1.JPG" /></p>

--------------------------------------------------------------------------------
<p align="center"><img width="80%" src="/images/intro_to_cv//mask_rcnn_ds.png" /></p>
<p align="center"><img width="80%" src="/images/intro_to_cv/maskrcnn_masks.png" /></p>
<p align="center"><img width="80%" src="/images/intro_to_cv/maskrcnn_bboxes.png" /></p>

--------------------------------------------------------------------------------


# Intro to Computer Vision Documentation

## Instructions: 

**installation of package**: 

`pip install pytorch-tutorials==0.2.19`

**example of how to use the below classes and methods**: 

*Note*: For more details on how to use the package, look at the end of the 
videos/notebooks in the computer vision [tutorial series](https://github.com/drewbyron/pytorch-tutorials/blob/main/README.md#intro-to-computer-vision).
```
# import
from pytorch_tutorials.intro_to_computer_vision import cv_utility
from pytorch_tutorials.intro_to_computer_vision import cv_datasets
from pytorch_tutorials.intro_to_computer_vision import cv_models
from pytorch_tutorials.intro_to_computer_vision import cv_pl_data_modules

# ------------------------Example 1: Datasets-------------------------------------------
# Grab a pytorch dataset for testing an object detection / image segmentation model. 
instance_seg_dataset = cv_pl_data_modules.ObjectDetection_DS(ds_size = 4, img_size = 256, shapes_per_image=(3,8), target_masks=True, rand_seed = 123456)

# ------------------------Example 2: Datamodules----------------------------------------

# Grab a torch lightning datamodule for testing an object detection / image segmentation model. 
instance_seg_dm = cv_pl_data_modules.ObjectDetection_DM(train_val_size = 1000, train_val_split = (.9,.1), test_size = 100, batch_size=4, img_size = 256, shapes_per_image=(3,8), target_masks=True, rand_seed = 123456)

# Grab a training batch.
instance_seg_dm.setup(stage = "fit")
dataiter = iter(instance_seg_dm.train_dataloader())
imgs, targets = dataiter.next()

# Turn batch tensor into list of images.
target_images = [img for img in imgs]

# Add target bounding boxes to images.
target_images = cv_utility.display_boxes(target_images, targets, instance_seg_dm.class_map, width=1, fill = True)

# Add target masks to images. 
target_images = display_masks_rcnn(target_images, targets, instance_seg_dm.class_map)

# Visualize.
grid = make_grid(target_images)
cv_utility.show(grid, figsize = (20, 20))

# ------------------------Example 2: Models---------------------------------------------

# Make a random image to test the model.
img_size = 16
batch_size = 4
x = torch.rand((batch_size, 3, img_size,img_size))

# Grab the model.
model = cv_models.get_maskrcnn(num_classes = 2, pretrained = True)

# Predict.
model.eval()
output = model(x)

# Look at output.
print(f"Input shape:\n {x.shape} \n" )
print(f"Mask RCNN Output (dict keys):\n {output[0].keys()}")
```

## Table of Contents  
- [cv_datasets](#cv_datasets)  
	- [Draw](#class-Draw)
	- [CV_DS_Base](#class-CV_DS_Base)
	- [ObjectCounting_DS](#class-ObjectCounting_DS)
	- [ImageSegmentation_DS](#class-ImageSegmentation_DS)
	- [ObjectDetection_DS](#class-ObjectDetection_DS)
- [cv_pl_data_modules](#cv_pl_data_modules)  
	- [ObjectCounting_DM](#class-ObjectCounting_DM)
	- [ImageSegmentation_DM](#class-ImageSegmentation_DM)
	- [ObjectDetection_DM](#class-ObjectDetection_DM)
- [cv_models](#cv_models)  
	- [DoubleConv](#class-DoubleConv)
	- [ObjectCounter](#class-ObjectCounter)
	- [UNET](#class-UNET)
	- [get_fasterrcnn](#def-get_fasterrcnn)
	- [get_maskrcnn](#def-get_maskrcnn)
- [cv_utility](#cv_utility)  
	- [show](#def-show)
	- [add_labels](#def-add_labels)
	- [labels_to_masks](#def-labels_to_masks)
	- [display_masks_unet](#def-display_masks_unet)
	- [display_boxes](#def-display_boxes)
	- [display_masks_rcnn](#def-display_masks_rcnn)
	- [display_labels](#def-display_labels)
	- [threshold_pred_masks](#def-threshold_pred_masks)
	- [build_coco_class_map](#def-build_coco_class_map)
	- [apply_score_cut](#def-apply_score_cut)
	- [load_img_dir](#def-load_img_dir)
	- [get_preds](#def-get_preds)
	- [save_imgs](#def-save_imgs)
	- [maskrcnn_process_images](#def-maskrcnn_process_images)
	- [maskrcnn_process_video](#def-maskrcnn_process_video)



## [cv_datasets](https://github.com/drewbyron/pytorch-tutorials/blob/main/pytorch_tutorials/intro_to_computer_vision/cv_datasets.py)

A set of pytorch datasets for building simple computer vision projects.

### class Draw

Class used to draw shapes onto images. Methods return coordinates of
corresponding shape on a 2d np array of shape (img_size, img_size).
The np rng is used for enabling derministic behaviour.

*Args:*

img_size (int): draws onto 2d array of shape (img_size, img_size).

rng (Generator): used for enabling deterministic behaviour. Example
    of valid rng: rng = np.random.default_rng(12345)


### class CV_DS_Base

Base class for a set of PyTorch computer vision datasets. This class
contains all of the attributes and methods common to all datasets
in this package.
Alone this base class has no functionality. The utility of these datasets
is that they enable the user to test cv models with very small and
simple images with tunable complexity. It also requires no downloading
of images and one can scale the size of the datasets easily.

*Args:*

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


### class ObjectCounting_DS

Self contained PyTorch Dataset for testing object counting models.

*Args:*

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


### class ImageSegmentation_DS

Self contained PyTorch Dataset for testing image segmentation models.

*Args:*

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


## class ObjectDetection_DS

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

*Args:*

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


## [cv_pl_data_modules](https://github.com/drewbyron/pytorch-tutorials/blob/main/pytorch_tutorials/intro_to_computer_vision/cv_pl_data_modules.py)

A set of pytorch lightning data modules for building simple computer vision projects.

### class ObjectCounting_DM

Self contained PyTorch Lightning DataModule for testing object
counting models with PyTorch Lightning.Uses the torch dataset
ObjectCounting_DS.

*Args:* 

train_val_size (int): total size of the training and validation
    sets combined.

train_val_split (Tuple[float, float]): should sum to 1.0. For example
    if train_val_size = 100 and train_val_split = (0.80, 0.20)
    then the training set will contain 80 imgs and the validation
    set will contain 20 imgs.

test_size (int): the size of the test data set.

batch_size (int): batch size to be input to dataloaders. Applies
    for training, val, and test datasets.

dataloader_shuffle (Dict): whether or not to shuffle for each of
    the three dataloaders. Dict must contain the keys: "train",
    "val", "test".

img_size (int): will build images of shape (3, img_size, img_size).

shapes_per_image (Tuple[int, int]): will produce images containing
    minimum number of shapes Tuple[0] and maximum number of shapes
    Tuple[1]. For example shapes_per_image = (2,2) would create a
    dataset where each image contains exactly two shapes.

class_probs (Tuple[float, float, float]): relative probability of
    each shape occuring in an image. Need not sum to 1. For example
    class_probs = (1,1,0) will create a dataset with 50% class 1
    shapes, 50% class 2 shapes, 0% class 3 shapes.

rand_seed (int): used to instantiate a numpy rng.

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
    build_imgs_and_targets() method of the ImageClassification_DS .



### class ImageSegmentation_DM

Self contained PyTorch Lightning DataModule for testing image
segmentation models with PyTorch Lightning. Uses the torch dataset
ImageSegmentation_DS.

*Args:*

train_val_size (int): total size of the training and validation
    sets combined.

train_val_split (Tuple[float, float]): should sum to 1.0. For example
    if train_val_size = 100 and train_val_split = (0.80, 0.20)
    then the training set will contain 80 imgs and the validation
    set will contain 20 imgs.

test_size (int): the size of the test data set.

batch_size (int): batch size to be input to dataloaders. Applies
    for training, val, and test datasets.

dataloader_shuffle (Dict): whether or not to shuffle for each of
    the three dataloaders. Dict must contain the keys: "train",
    "val", "test".

img_size (int): will build images of shape (3, img_size, img_size).

shapes_per_image (Tuple[int, int]): will produce images containing
    minimum number of shapes Tuple[0] and maximum number of shapes
    Tuple[1]. For example shapes_per_image = (2,2) would create a
    dataset where each image contains exactly two shapes.

class_probs (Tuple[float, float, float]): relative probability of
    each shape occuring in an image. Need not sum to 1. For example
    class_probs = (1,1,0) will create a dataset with 50% class 1
    shapes, 50% class 2 shapes, 0% class 3 shapes.

rand_seed (int): used to instantiate a numpy rng.

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


### class ObjectDetection_DM

Self contained PyTorch Lightning DataModule for testing object detection
and image segmentation models with PyTorch Lightning. Uses the torch
dataset ObjectDetection_DS.

*Args:* 

train_val_size (int): total size of the training and validation
    sets combined.

train_val_split (Tuple[float, float]): should sum to 1.0. For example
    if train_val_size = 100 and train_val_split = (0.80, 0.20)
    then the training set will contain 80 imgs and the validation
    set will contain 20 imgs.

test_size (int): the size of the test data set.

batch_size (int): batch size to be input to dataloaders. Applies
    for training, val, and test datasets.

dataloader_shuffle (Dict): whether or not to shuffle for each of
    the three dataloaders. Dict must contain the keys: "train",
    "val", "test".

img_size (int): will build images of shape (3, img_size, img_size).

shapes_per_image (Tuple[int, int]): will produce images containing
    minimum number of shapes Tuple[0] and maximum number of shapes
    Tuple[1]. For example shapes_per_image = (2,2) would create a
    dataset where each image contains exactly two shapes.

class_probs (Tuple[float, float, float]): relative probability of
    each shape occuring in an image. Need not sum to 1. For example
    class_probs = (1,1,0) will create a dataset with 50% class 1
    shapes, 50% class 2 shapes, 0% class 3 shapes.

rand_seed (int): used to instantiate a numpy rng.

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

## [cv_models](https://github.com/drewbyron/pytorch-tutorials/blob/main/pytorch_tutorials/intro_to_computer_vision/cv_models.py)

A set of pytorch computer vision models or functions to get a pretrained model with a custom number of output classes. 

### class DoubleConv

A double convolution module used to extract features.

*Args:*

in_channels (int): number of input channels. For example for an
    input of shape (batch_size, 3, img_size, img_size) in_channels
    is 3.

out_channels (int): number of output_channels desired. For example
    if the desired output shape is (batch_size, 3, img_size, img_size)
    in_channels is 3.

kernel_size (int): A kernel of shape (kernel_size, kernel_size)
    will be applied to the imgs during both Conv2d layers.

bias (bool): whether or not to add a bias to the Conv2d layers.


### class ObjectCounter

An object counting model that uses multiple conv layers and then
two fully connected layers to determine how many instances of different
classes of objects are in an image.

*Args:*

img_size (int): model will take images of shape
    (3, img_size, img_size).

in_channels (int): number of input channels. For example for an
    put of shape (batch_size, 3, img_size, img_size) in_channels
    is 3.

num_classes (int): number of output classes desired. The output
    shape of the model will be (batch_size, num_classes).

features (List[int]): A list specifying the number of features to
    be used in each DoubleConv layer. Note that for the model to
    work the image_size must be divisable by {(2** len(features))}.

fc_intermediate_size (int): Size of the output of the first
    fully connected layer (fc1) and size of the input of the second
    fully connected layer (fc2).

kernel_size (int): A kernel of shape (kernel_size, kernel_size)
    will be applied to the imgs during both Conv2d layers.

bias (bool): whether or not to add a bias to the Conv2d layers.


### class UNET
A PyTorch implimentation of a UNET image segmentation model based
on this work: https://arxiv.org/abs/1505.04597. Specifics based on
Aladdin Persson's implimentation:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py

*Args:*

in_channels (int): number of input channels. For example for an
	put of shape (batch_size, 3, img_size, img_size) in_channels
	is 3.

num_classes (int): number of output classes desired. The output
    shape of the model will be (batch_size, num_classes, img_size,
    img_size). For example output[0][i] is a binary segmentation
    mask for class i. Note that class 0 is reserved for background.

first_feature_num (int): An int specifying the number of features to
    be used in the first DoubleConv layer.

num_layers (int): Number of layers to use in the UNET architecture.
    The ith layer contains first_feature_num * 2**i features. Note 
    that if img_size // 2**num_layers < 1 then the model will break.

kernel_size (int): A kernel of shape (kernel_size, kernel_size)
    will be applied to the imgs during both Conv2d layers of
    DoubleConv.

bias (bool): whether or not to add a bias to the DoubleConv Conv2d
    layers.

### def get_fasterrcnn

A function for loading the PyTorch implimentation of FasterRCNN.
To not have predictor changed at all set num_classes = -1.
See here for documentation on the input and output specifics:
https://pytorch.org/vision/stable/models/faster_rcnn.html

*Args:* 

num_classes (int): number of output classes desired.

pretrained (bool): whether or not to load a model pretrained on the COCO dataset. 

*Returns:*

model (nn.Module): torchvision faster rcnn implimentation with custom 
	number of outputs/classes.


### def get_maskrcnn

A function for loading the PyTorch implimentation of MaskRCNN.
To not have predictor changed at all set num_classes = -1.
See here for documentation on the input and output specifics:
https://pytorch.org/vision/0.12/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html

*Args:*

num_classes (int): number of output classes desired.

pretrained (bool): whether or not to load a model pretrained on the COCO dataset. 

*Returns:*

model (nn.Module): torchvision mask rcnn implimentation with custom
    number of outputs/classes.


## [cv_utility](https://github.com/drewbyron/pytorch-tutorials/blob/main/pytorch_tutorials/intro_to_computer_vision/cv_utility.py)

Utility functions for pytorch computer vision tasks.

### def show

Displays a single image or list of images. Taken more or less from
the pytorch docs:
https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html#visualizing-a-grid-of-images

*Args:*

imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
    of shape (3, H, W) or a single image of shape (3, H, W).

figsize (Tuple[float, float]): size of figure to display.

*Returns:*

None


### def add_labels

Takes a single image of shape (3, H, W) and adds labels directly
onto the image using cv2. Used with ImageSegmentation_DS/DM but can
be used in other applicable computer vision tasks.

*Args:*
   
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


*Returns:*

img (torch.UInt8Tensor[3, H, W]): a pytorch image with the names
    and (optionally) counts corresponding to the provided label
    drawn over the image.


### def labels_to_masks

Converts  a batch of segmentation labels into binary masks. Used
with UNET or in other image segmentation tasks. This function works
for both batches of labels or single (2d) image labels. The Args and
return descriptions assume a full batch is input.

*Args:*

labels (torch.int64[batch_size, H, W]): a batch of segmentation
    labels. Each pixel is assigned a class (an integer value).

*Returns:*

binary_masks (torch.bool[batch_size, num_obj_ids, H, W]): a batch of
    corresponding binary masks. Layer i (of dim = 1) corresponds to
    a binary mask for class i. The total number of binary masks will
    be the number of unique object ids (num_obj_ids).



### def display_masks_unet

Takes a batch of images and a batch of masks of the same length and
overlays the images with the masks using the "target_color" specified
in the class_map.

*Args:* 

imgs (List[torh.ByteTensor[batch_size, 3, H, W]]): a batch of
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

*Returns:*

result_imgs (List[torch.ByteTensor[3, H, W]]]): list of images
    with overlaid segmentation masks.


### def display_boxes
Takes a list of images and a list of target or prediction dictionaries
of the same len and overlays bounding boxes onto the images.

*Args:*

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

*Returns:*

result_imgs (List[torch.ByteTensor[3, H, W]]): list of images with
    overlaid bounding boxes.


### def display_masks_rcnn

Takes a list of images and a list of target or prediction dictionaries
of the same len and overlays segmentation masks onto the images.

*Args:*

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

*Returns:*

result_imgs (List[torch.ByteTensor[3, H, W]]): list of images with
    overlaid segmentation masks.



### def display_labels

Takes a list of images and a list of target or prediction dictionaries
of the same len and adds labels to the instances. Note that for very
small images this will behave poorly.

*Args:* 

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

*Returns:*

labeled_imgs (List[torch.ByteTensor[3, H, W]]): list of images with
    overlaid instance labels.



### def threshold_pred_masks

Takes a list of prediction dictionaries (one for each image) and
thresholds the soft masks, returning a list of prediction dictionaries
with thresholded (boolean) masks.

*Args:*

preds (List[Dict[torch.Tensor]]): predictions as output by the
    torchvision implimentation of MaskRCNN. The masks consist of
    probabilities (torch.float32) in the range (0,1) for each pixel.
    See link below for details on the target/prediction formatting.
    https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

*Returns:*

thresholded_preds (List[Dict[torch.Tensor]]): predictions with
    boolean (torch.bool) masks.


### def build_coco_class_map

Returns a class_map for coco classes.

*Args:*

seed (int): seed to use to build np rng. Can change this to get
    a new set of colors.

drop_background (bool): If true the background class (assigned
    class_id = 0 by default) will be dropped from the class map.
    This is the default because in displaying the segmented images
    often one doesn't care to display background.

*Returns:*

coco_class_map (Dict[Dict]): class_map to be used with other functions
    in this module.



### def apply_score_cut

Takes a list of prediction dictionaries (one for each image) and cuts
out all instances whose score is below the score threshold.

*Args:*

preds (List[Dict[torch.Tensor]]): predictions as output by the
    torchvision implimentation of MaskRCNN or FasterRCNN. The 
    scores are in the range (0,1) and signify the certainty of 
    the model for that instance.
    See link below for details on the target/prediction formatting.
    https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html

score_threshold (float): the threshold to apply to the identified
    objects. If an instance is below the score_threshold it will
    be removed from the score_thresholded_preds dictionary.

*Returns:*

score_thresholded_preds (List[Dict[torch.Tensor]]): predictions
    that exceed score_threshold.


### def load_img_dir

Loads all of the images in a directory into torch images.

*Args:*

path (str): path should point to a directory that only contains
    .JPG images. Or any image type compatible with cv2.imread().

resize_factor (float): how to resize the image. Often one would
    like to reduce the size of the images to be easier/faster to
    use with our maskrcnn model.

*Returns:*

imgs (List[torch.ByteTensor[3, H, W]]): list of images (each a
    torch.ByteTensor of shape(3, H, W)).


### def get_preds

Simple utility function for returning the predictions of maskrcnn model.
This deals with putting the model and imgs on device and normalizing
torch.ByteTensor.

*Args:*

maskrcnn (nn.Module): an instance of the torchvision Mask RCNN
    model. One can build with following call: maskrcnn =
    cv_models.get_maskrcnn(num_classes=-1, pretrained=True)

device (str): what device to put model and imgs. Use following
    call: device = "cuda" if torch.cuda.is_available() else "cpu"

*Returns:*

preds (List[Dict[torch.Tensor]]): predictions as output by the
    torchvision implimentation of MaskRCNN.
    See link below for details on the target/prediction formatting.
    https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/mask_rcnn.html


### def save_imgs

Saves torch images to JPG file format.

*Args:*

imgs (List[torch.ByteTensor[3, H, W]]): list of images (each a
    torch.ByteTensor of shape(3, H, W)).

base_path (str): path to directory where images should be written.

base_name (str): base name to be used to build JPG file paths.

*Returns:*

None


### def maskrcnn_process_images


Processes a set of imgs and associated predictions.

*Args:*

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

*Returns:*

processed_imgs (List[torch.ByteTensor[3, H, W]]): list of processed
    images (each a torch.ByteTensor of shape(3, H, W)).


### def maskrcnn_process_video

Processes a .MOV, adding segmentation, labels, and/or bboxes and
writes out a processed version of the original .MOV

*Args:*

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

frame_max (int): number of frames to process. Used to limit the time this takes and to sanity check things. Set to 10 to be sure things are working.

fps (int): frames per second of video. Default for google photos
is 30.

*Returns:*

None
"""