"""The network definition that was used for a second place solution at the DeepGlobe Building Detection challenge."""

import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import Sequential
from collections import OrderedDict
# ABN requires "Ninja is required to load C++ extensions"
from modules.bn import ABN

from modules.wider_resnet import WiderResNet


def conv3x3(in_, out):
    """
    Creates 3x3 conv filter, with stride 1 and padding 1
    :param in_:
    :param out:
    :return:
    """
    # stride default is 1
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    """
    Custom nn module with Convolution and Relu afterwards.
    """
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=False):
        '''
        creates the decoder block which (by default is_conv=False) consists of usampling + 2 deconv layers
        :param in_channels:
        :param middle_channels: channels for first conv layer
        :param out_channels:
        :param is_deconv:
        '''
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                # upsample with nearest-neighbour method, scale up D,W,H by factor 2
                nn.Upsample(scale_factor=2, mode='nearest'),
                # 2x (ConvRelu)
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)
            )

    def forward(self, x):
        return self.block(x)


class TernausNetV2(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""

    def __init__(self, num_classes=1, num_filters=32, is_deconv=False, num_input_channels=11, **kwargs):
        """

        Args:
            num_classes: Number of output classes.
            num_filters:
            is_deconv:
                True: Deconvolution layer is used in the Decoder block.
                False: Upsampling layer is used in the Decoder block. (default setting in paper)
            num_input_channels: Number of channels in the input images.
        """
        super(TernausNetV2, self).__init__()

        if 'norm_act' not in kwargs:
            norm_act = ABN
        else:
            norm_act = kwargs['norm_act']

        self.pool = nn.MaxPool2d(2, 2)

        # setup widerResnet as encoder,
        # structure defines number of residual blocks, we define here 6 blocks! paper talks about 5 blocks
        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], classes=1000, norm_act=norm_act)

        # define first layer. Why not bias?
        self.conv1 = Sequential(
            OrderedDict([('conv1', nn.Conv2d(num_input_channels, 64, 3, padding=1, bias=False))]))

        # use 2nd to 5th module of WideResnet (module consists of multiple blocks)
        self.conv2 = encoder.mod2 # 128 channels, 3 bottleneck blocks, each with with `1 x 1`, then `3 x 3` then `1 x
        # 1` convolutions.
        self.conv3 = encoder.mod3 # 256 channels, blocks as above
        self.conv4 = encoder.mod4 # 512 channels, as above
        self.conv5 = encoder.mod5 # 1024 channels, as above


        # center block
        self.center = DecoderBlock(1024, num_filters * 8, num_filters * 8, is_deconv=is_deconv) # 256 output

        # rest of the decoder blocks, input channels given through encoder input + num_filters*8
        #                                                                                               (encoder input dim, decoder input)
        self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv) # (1024,256) input channels
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv) # (512,256) input channels
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv) # (256,256) input ..
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv) # (128,64) input

        # simple convolution, 3x3
        self.dec1 = ConvRelu(64 + num_filters, num_filters) #(64,32)

        # final output layer for number of classes given, 1x1 convolution to get two
        # # channels, binary mask and for touching instance
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):

        # encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        # decoders with concatunated encoder part
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.final(dec1)
