<p align="center"><img width="15%" src="/images/logos/pytorch_logo.png" /></p>

--------------------------------------------------------------------------------

<p align="center"><img width="50%" src="/images/logos/pytorch_tutorials_logo.png" /></p>

--------------------------------------------------------------------------------

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://docs.python.org/3.9/)
[![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/docs/stable/index.html)
[![](https://img.shields.io/badge/PyTorchLightning-792EE5?style=for-the-badge&logo=PyTorchLightning&logoColor=white)](https://pytorch-lightning.readthedocs.io/en/stable/)
[![Open In Youtube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/channel/UCORZQS8pVWrPyY3-OpvNkcg) 
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
	- [Part 4: Object Detection with Mask RCNN](#Object-Detection-with-Mask-RCNN)
	- [Part 5: Applying Mask RCNN in the Wild](#Applying-Mask-RCNN-in-the-Wild)
- [How to Reach Me](#How-to-Reach-Me) 


## Summary 

In this repository you will find tutorials aimed at helping people get up to speed with pytorch and pytorch lightning. Links to the relavent docs and associated youtube channel and pypi project can be found in the badges above. For each video in the youtube series there is an associated colab notebook (links below). 

My ultimate intent is to have a little series for all of the pytorch libraries (vision, text, audio,..) but for now it's just vision. 

I hope this is useful to you! Please comment on youtube if you have corrections/comments or suggestions for how to make this project and associated package more useful. Thanks!


## Intro to Computer Vision

This tutorial series consists of five youtube videos and four notebooks. We begin by building a model to count the number of shapes in an image, building incrimentally towards training Faster RCNN and Mask RCNN. We finish with some examples of how to apply this model to real images and videos. The series will be most useful to those who have some pytorch and machine learning experience and are intersted in deep learning for computer vision tasks. 

In my experience it is very difficult to build intuition for how deep learning works in practice without actually trying to train models. So my hope is that this will be a good way for people to get their hands dirty without a large barrier to entry in set-up or prior experience. I highly encourage people to make a copy of the colab notebooks you can edit and follow along with it while watching the videos. My idea with this project was to make small configurable datasets that would enable people (and myself) to gain intuition for how to train bigger models (FasterRCNN, MaskRCNN,...) without needing to run on a cluster or train for a week to see how things work. The idea is that the youtube videos and associated colab notebooks walk through everything in detail (hopefully not excruciating detail) and then the classes and utility functions we make in the colab notebooks are all found in the [pytorch-tutorials pypi](https://pypi.org/project/pytorch-tutorials/) package so that people can easily make their own sandbox to play around in. 

### Summary and Introduction

* [![][youtube_logo]](https://www.youtube.com/channel/UCORZQS8pVWrPyY3-OpvNkcg)
* [Intro to Computer Vision Documentation](/pytorch_tutorials/intro_to_computer_vision/README.md)

### Object Counting with CNNs

* [![][youtube_logo]](https://www.youtube.com/channel/UCORZQS8pVWrPyY3-OpvNkcg)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Cb9jxZ75Svivcxk2Qd_Y8o5PccCWLM-A?usp=sharing)

### Image Segmentation with UNET

* [![][youtube_logo]](https://www.youtube.com/channel/UCORZQS8pVWrPyY3-OpvNkcg)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/136quz_ISw1b5sOj7wUC2WiVQca6iR-wR?usp=sharing)

### Object Detection with Faster RCNN

* [![][youtube_logo]](https://www.youtube.com/channel/UCORZQS8pVWrPyY3-OpvNkcg)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IGhNjysjbbchjQO19ta-frpNG3DDS_uV?usp=sharing)

### Object Detection with Mask RCNN

* [![][youtube_logo]](https://www.youtube.com/channel/UCORZQS8pVWrPyY3-OpvNkcg)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bIU6kMxGtzcr5vnBxkM3bWyEaHZJzepD?usp=sharing)

### Applying Mask RCNN in the Wild

* [![][youtube_logo]](https://www.youtube.com/channel/UCORZQS8pVWrPyY3-OpvNkcg)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YaM6HN3IlLLy5tzWN7QCPi3ayKrIMJUp?usp=sharing)


## How to Reach Me

* If you have any suggestions for how to make this resource more useful please comment on the youtube channel. 
* If something in the package is broken please submit an [issue](https://github.com/drewbyron/pytorch-tutorials/issues) on github.
* For work stuff or to just say hi reach out over [linked-in](www.linkedin.com/in/drew-byron). 

--------------------------------------------------------------------------------
[![](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/drew-byron)
[![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/drewbyron/pytorch-tutorials)

--------------------------------------------------------------------------------
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


--------------------------------------------------------------------------------
*Note: This project is not affiliated with pytorch in any way.*
