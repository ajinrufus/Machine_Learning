'''
Functionality to instantiate TinyVGG model'''

import torch
from torch import nn

class TinyVGG(nn.Module):
    '''Creates TinyVGG Architecture by replicating https://poloclub.github.io/cnn-explainer/
    Args:
        in_channels : number of input channels (eg: 3 for RGB)
        hid_units   : number of hidden channels or nodes for each layer
        out_classes : number of classes for classification
        image_shape : height or width of image

    Creates:
        model instance and can predictions'''

    def __init__(self, in_channels: int, hid_units: int,
                 out_classes: int, image_shape: int):
        super().__init__()

        PADDING, STRIDE, KERNEL, DILATION = 1, 2, 3, 1 # dilation: 1 is default

        # 1st block
        self.conv_block_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                    out_channels=hid_units,
                                                   kernel_size=KERNEL, stride=1,
                                                   padding=PADDING),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=hid_units,
                                                    out_channels=hid_units,
                                                    kernel_size=KERNEL, stride=STRIDE,
                                                    padding=PADDING),
                                          nn.ReLU())
        
        
        # required to compute input shape of classifier layer
        H_out, W_out = H_W_out(image_shape, PADDING, KERNEL, STRIDE, DILATION)

        # 2nd block
        self.conv_block_2 = nn.Sequential(nn.Conv2d(in_channels=hid_units,
                                                    out_channels=hid_units,
                                                    kernel_size=KERNEL, stride=1,
                                                    padding=PADDING),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=hid_units,
                                                    out_channels=hid_units,
                                                    kernel_size=KERNEL, stride=STRIDE,
                                                    padding=PADDING),
                                          nn.ReLU())
        
        H_out, W_out = H_W_out(H_out, PADDING, KERNEL, STRIDE, DILATION)
        
        # output classifier layer
        self.classifier = nn.Sequential(nn.Flatten(), 
                                        nn.Linear(in_features=hid_units*H_out*W_out,
                                                  out_features=out_classes))

    
    def forward(self,x):

        # operating w/o storing is faster - operation fusion
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def H_W_out(image_shape, pad, kernel, stride, dilation=1):
    '''Output data shape after convolution

    Args:
        image_shape : height or width of input image
        pad         : amount of padding required
        kernel      : kernel size
        stride      : number of pixels to be skipped during convolution
        dilation    : involved with kernel, default is 1

    returns:
        tuple of expected output height and width of image after covolution
        H_out : output height of image
        W_out : output width of image'''

    H_out = int((image_shape + 2*pad - dilation * (kernel -1) -1)/stride +1) # output height of data
    W_out = int((image_shape + 2*pad - dilation * (kernel -1) -1)/stride +1) # output width of data
    return H_out, W_out
