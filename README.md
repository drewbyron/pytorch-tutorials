<p align="center"><img width="15%" src="/images/logos/pytorch_logo.png" /></p>

--------------------------------------------------------------------------------

<p align="center"><img width="50%" src="/images/logos/pytorch_tutorials_logo.png" /></p>

--------------------------------------------------------------------------------

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://docs.python.org/3.9/)
[![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/docs/stable/index.html)
[![](https://img.shields.io/badge/PyTorchLightning-792EE5?style=for-the-badge&logo=PyTorchLightning&logoColor=white)](https://pytorch-lightning.readthedocs.io/en/stable/)
[![Open In Youtube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/channel/UCORZQS8pVWrPyY3-OpvNkcg/playlists) 
[![](https://img.shields.io/badge/pypi-3775A9?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/pytorch-tutorials/)

--------------------------------------------------------------------------------

[youtube_logo]: /images/logos/youtube_logo.png

# pytorch-tutorials

### Table of Contents  
- [Summary](#Summary)  
- [Intro to Computer Vision](#Intro-to-Computer-Vision)  
	- [Part 0: Summary and Introduction](#Summary-and-Introduction)
	- [Part 1: Object Counting with CNNs](#Object-Counting-with-CNNs)
	- [Part 2: Image Segmentation with UNET](#Image-Segmentation-with-UNET)
	- [Part 3: Object Detection with Faster RCNN](#Object-Detection-with-Faster-RCNN)
	- [Part 4: Instance Segmentation with Mask RCNN](#Instance-Segmentation-with-Mask-RCNN)
	- [Part 5: Applying Mask RCNN in the Wild](#Applying-Mask-RCNN-in-the-Wild)
- [How to Reach Me](#How-to-Reach-Me) 


## Summary 

In this repository, you will find tutorials aimed at helping people get up to speed with PyTorch and PyTorch Lightning. Links to the relevant docs and associated youtube channel and PyPI project can be found in the badges above. My ultimate intent is to have a little series associated with all of the PyTorch libraries (vision, text, audio,..) but for now it's just vision. 

I hope this is useful to you! Please comment on youtube if you have corrections/comments or suggestions for how to make this project and associated package more useful. Thanks!


## Intro to Computer Vision

This tutorial series consists of six youtube videos (one intro and five tutorials) and five public colab notebooks so you can follow along with the video. We begin by building a model to count objects in an image, then conduct image segmentation using UNET, and finally, learn how to train Faster RCNN and Mask RCNN on a custom dataset. We work with the same simple and configurable dataset of images throughout the series. We finish with some examples of how to apply this model to real images and videos in Part 5. The series will be most useful to those who have some PyTorch and machine learning experience and are interested in deep learning for computer vision tasks. We will also be using Pytorch Lightning extensively which is a great tool for organizing deep learning projects so if you are new to that tool, this series will be a good primer and tour of the features on offer. Each tutorial contains a video and a colab notebook so you can follow along with the video and easily do your own explorations.

In my experience, it is challenging to build intuition for how deep learning works in practice without actually trying to train models. So my hope is that this will be a good way for people to get their hands dirty without a large barrier to entry in set-up or prior experience. I highly encourage people to make a copy of the colab notebooks you can edit (open the notebook by right-clicking the badges below, then click "save a copy in drive") and follow along with it while watching the videos. My idea with this project was to make small configurable datasets that would enable people (and myself) to gain an intuition for how to train bigger models (FasterRCNN, MaskRCNN,...) without needing to run on a cluster or train for a week to see how things work. The idea is that the youtube videos and associated colab notebooks walk through everything in detail (hopefully not excruciating detail) and then the classes and utility functions we make in the colab notebooks are all found in the [pytorch-tutorials pypi](https://pypi.org/project/pytorch-tutorials/) package so that people can easily make their own sandbox to play around in. Documentation for that package is found [here](/pytorch_tutorials/intro_to_computer_vision/README.md).

**TLDR:** By the end of the series, a PyTorch computer vision novice should have the tools to train any of the models we cover on a custom dataset (Part 1 - Part 4) and also quickly apply a trained mask rcnn model to their own images and videos (Part 5). 


### Part 0: Summary and Introduction <a name="Summary-and-Introduction"></a>

* **-> Start Here <-**
* [![][youtube_logo]](https://youtu.be/dsSMv7YB5B0)
* [Intro to Computer Vision Documentation](/pytorch_tutorials/intro_to_computer_vision/README.md)

### Part 1: Object Counting with CNNs <a name="Object-Counting-with-CNNs"></a>

* [![][youtube_logo]](https://youtu.be/VLJ6RNlvS3Y)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Cb9jxZ75Svivcxk2Qd_Y8o5PccCWLM-A?usp=sharing)

### Part 2: Image Segmentation with UNET <a name="Image-Segmentation-with-UNET"></a>

* [![][youtube_logo]](https://youtu.be/-I3yThvtkHs)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/136quz_ISw1b5sOj7wUC2WiVQca6iR-wR?usp=sharing)

### Part 3: Object Detection with Faster RCNN <a name="Object-Detection-with-Faster-RCNN"></a>

* [![][youtube_logo]](https://youtu.be/X50peXk34fU)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IGhNjysjbbchjQO19ta-frpNG3DDS_uV?usp=sharing)

### Part 4: Instance Segmentation with Mask RCNN <a name="Instance-Segmentation-with-Mask-RCNN"></a>

* [![][youtube_logo]](https://youtu.be/2w7Enf0Lo9A)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bIU6kMxGtzcr5vnBxkM3bWyEaHZJzepD?usp=sharing)

### Part 5: Applying Mask RCNN in the Wild <a name="Applying-Mask-RCNN-in-the-Wild"></a>

* [![][youtube_logo]](https://youtu.be/vG8s7BAjChc)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YaM6HN3IlLLy5tzWN7QCPi3ayKrIMJUp?usp=sharing)
* [Images and Videos used in tutorial](https://drive.google.com/drive/folders/1-Hx9D18T2HAqmsuFrYzCEWWlpCrcXMEH?usp=sharing
)


## How to Reach Me

* If you have any suggestions for how to make this resource more useful please comment on the youtube channel. 
* If something in the package is broken please submit an [issue](https://github.com/drewbyron/pytorch-tutorials/issues) on GitHub.
* For work stuff or to just say hi reach out over [linked-in](www.linkedin.com/in/drew-byron). 

--------------------------------------------------------------------------------
[![](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/drew-byron)
[![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/drewbyron/pytorch-tutorials)

--------------------------------------------------------------------------------
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


--------------------------------------------------------------------------------
*Note: This project is not affiliated with pytorch in any way.*
