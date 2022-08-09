# Author: Drew Byron.
# Date: 7/23/22.
# Description of Module:

# Deep learning imports.
import torch
from torch import nn

import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader, random_split, TensorDataset

from torchmetrics.detection.mean_ap import MeanAveragePrecision


class DoubleConv(nn.Module):
    """
    A double convolution module used to extract features.

    Args:
        in_channels (int): number of input channels. For example for an
            input of shape (batch_size, 3, img_size, img_size) in_channels
            is 3.
        out_channels (int): number of output_channels desired. For example
            if the desired output shape is (batch_size, 3, img_size, img_size)
            in_channels is 3.
        kernel_size (int): A kernel of shape (kernel_size, kernel_size)
            will be applied to the imgs during both Conv2d layers.
        bias (bool): whether or not to add a bias to the Conv2d layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding="same",
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding="same",
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ObjectCounter(nn.Module):
    """An object counting model that uses multiple conv layers and then
    two fully connected layers to determine how many instances of different
    classes of objects (shapes) are in an image.

    Args:
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
    """

    def __init__(
        self,
        img_size=256,
        in_channels=3,
        num_classes=3,
        features=[16, 32],
        fc_intermiate_size=10,
        kernel_size=3,
        bias=True,
    ):

        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.fc_intermiate_size = fc_intermiate_size
        self.kernel_size = kernel_size
        self.bias = True

        final_size = self.img_size / (2 ** len(self.features))

        if (final_size % 1) != 0:
            raise ValueError(f"image_size must be divisable by {(2** len(features))}.")

        self.final_size = int(final_size)
        self.final_feature_size = self.features[-1]
        self.fc_in_size = self.final_feature_size * self.final_size**2

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.fc_in_size, self.fc_intermiate_size)
        self.fc2 = nn.Linear(self.fc_intermiate_size, self.num_classes)
        self.feature_extractor = nn.ModuleList()

        # Feature extractor
        for feature in features:
            self.feature_extractor.append(
                DoubleConv(
                    in_channels, feature, kernel_size=self.kernel_size, bias=self.bias
                )
            )
            in_channels = feature

    def forward(self, x):

        for module in self.feature_extractor:
            x = module(x)
            x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class UNET(nn.Module):
    """A PyTorch implimentation of a UNET image segmentation model based
    on this work: https://arxiv.org/abs/1505.04597. Specifics based on
    Aladdin Persson's implimentation:
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py

    Args:
        in_channels (int): number of input channels. For example for an
            put of shape (batch_size, 3, img_size, img_size) in_channels
            is 3.
        num_classes (int): number of output classes desired. The output
            shape of the model will be (batch_size, num_classes, img_size,
            img_size). For example output[0][i] is a binary segmentation
            mask for class i. Note that class 0 is reserved for background.
        features (List[int]): A list specifying the number of features to
            be used in each DoubleConv layer. Note that for the model to
            work the image_size must be divisable by {(2** len(features))}.
        kernel_size (int): A kernel of shape (kernel_size, kernel_size)
            will be applied to the imgs during both Conv2d layers of
            DoubleConv.
        bias (bool): whether or not to add a bias to the DoubleConv Conv2d
            layers.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=4,
        features=[64, 128, 256, 512],
        kernel_size=3,
        bias=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.kernel_size = kernel_size
        self.bias = bias

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], self.num_classes, kernel_size=1)

        # Down part of UNET.
        for feature in self.features:
            self.downs.append(
                DoubleConv(
                    in_channels, feature, kernel_size=self.kernel_size, bias=self.bias
                )
            )
            in_channels = feature

        # Up part of UNET.
        for feature in reversed(self.features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(
                DoubleConv(
                    feature * 2, feature, kernel_size=self.kernel_size, bias=self.bias
                )
            )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        return x
