import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    创建一个卷积层，后跟 batchNorm 和 relu。Bias 为 False，因为 batchnorm 无论如何都会使其无效。Args

    :
        in_channels (int) : 卷积层的输入通道
        out_channels (int) : 卷积层的输出通道
        kernel_size (int) : 过滤器大小
        stride (int) : 卷积过滤器移动的像素数
        padding (int) : 边界周围的额外零像素，影响输出特征图的大小


    属性：
        由 conv->batchnorm->relu 组成的层

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvBlock, self).__init__()
        # 2d 卷积
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)

        # batchnorm
        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        # relu 层
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batchnorm2d(self.conv2d(x)))


def testConvBlock():
    x = torch.randn(64, 1, 28, 28)
    model = ConvBlock(1, 3, 3, 1, 1)
    print(model(x).shape)
    del model


testConvBlock()
