<p align="center"><img width="15%" src="/images/logos/pytorch_logo.png" /></p>

--------------------------------------------------------------------------------

<p align="center"><img width="50%" src="/images/logos/pytorch_tutorials_logo_cv.png" /></p>

--------------------------------------------------------------------------------

<p align="center"><img width="40%" src="/images/intro_to_cv/mt_rainier_seg_0.JPG" /><img width="40%" src="/images/intro_to_cv/mt_rainier_seg_1.JPG" /></p>

--------------------------------------------------------------------------------
<p align="center"><img width="80%" src="/images/intro_to_cv//mask_rcnn_ds.png" /></p>
<p align="center"><img width="80%" src="/images/intro_to_cv/maskrcnn_masks.png" /></p>
<p align="center"><img width="80%" src="/images/intro_to_cv/maskrcnn_bboxes.png" /></p>

--------------------------------------------------------------------------------


--------------------------------------------------------------------------------

# Intro to Computer Vision Documentation


### Table of Contents  
[cv_datasets](#datasets)  
[cv_pl_data_modules](#pl_data_modules)  
[cv_models](#models)  
[cv_utility](#utility)  


<a name="datasets"/>

## cv_datasets

--- 
**class Draw:**

Class used to draw shapes onto images. Methods return coordinates of
corresponding shape on a 2d np array of shape (img_size, img_size).
The np rng is used for enabling derministic behaviour.

Args:
img_size (int): draws onto 2d array of shape (img_size, img_size).
rng (Generator): used for enabling deterministic behaviour. Example
    of valid rng: rng = np.random.default_rng(12345)

---
**class CV_DS_Base(torch.utils.data.Dataset):**

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

---

<a name="pl_data_modules"/>

## cv_pl_data_modules

<a name="models"/>

## cv_models

<a name="utility"/>

## cv_utility

