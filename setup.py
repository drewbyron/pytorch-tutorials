from setuptools import setup, find_packages

setup(
    name='pytorch_tutorials',
    version='0.2.19',    
    description='A resource for learning about PyTorch and deep learning.',
    long_description=
    "## [Project Documentation](https://github.com/drewbyron/pytorch-tutorials/blob/main/README.md)\n\n ## [Project YouTube Channel](https://www.youtube.com/channel/UCORZQS8pVWrPyY3-OpvNkcg/featured)\n\n",
    long_description_content_type='text/markdown',
    url='https://github.com/drewbyron/pytorch-tutorials',
    author='William (Drew) Byron',
    author_email='william.andrew.byron@gmail.com',
    license='MIT License',
    packages=find_packages(),
    install_requires= [
        "torch >= 1.12.1",
        "torchvision >= 0.13.1",
        "pytorch-lightning >= 1.6.4",
        "torchmetrics >= 0.9.1",
        "matplotlib>=3.1.3",
        "numpy>= 1.21.6",
        "opencv-python"
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        "Operating System :: OS Independent"
    ],
)