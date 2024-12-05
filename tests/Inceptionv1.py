import torch
from torch import nn
from torch.fx.experimental.fx_acc.acc_utils import draw_graph

from tests.ConvBlock import ConvBlock
from tests.InceptionBlock import InceptionBlock


class Inceptionv1(nn.Module):
    '''
    step-by-step building the inceptionv1 architecture. Using testInceptionv1 to evaluate the dimensions of output after each layer and deciding the padding number.

    Args:
        in_channels (int) : input channels. 3 for RGB image
        num_classes : number of classes of training dataset

    Attributes:
        inceptionv1 model

    For conv2 2 layers with first having 1x1 conv
    '''

    def __init__(self, in_channels, num_classes):
        super(Inceptionv1, self).__init__()

        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(ConvBlock(64, 64, 1, 1, 0), ConvBlock(64, 192, 3, 1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # in_channels , out_1x1 , red_3x3 , out_3x3 , red_5x5 , out_5x5 , out_1x1_pooling
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        print('conv1', x.shape)
        x = self.maxpool1(x)
        print('maxpool1', x.shape)

        x = self.conv2(x)
        print('conv2', x.shape)
        x = self.maxpool2(x)
        print('maxpool2', x.shape)

        x = self.inception3a(x)
        print('3a', x.shape)

        x = self.inception3b(x)
        print('3b', x.shape)

        x = self.maxpool3(x)
        print('3bmax', x.shape)

        x = self.inception4a(x)
        print('4a', x.shape)

        x = self.inception4b(x)
        print('4b', x.shape)

        x = self.inception4c(x)
        print('4c', x.shape)

        x = self.inception4d(x)
        print('4d', x.shape)

        x = self.inception4e(x)
        print('4e', x.shape)

        x = self.maxpool4(x)
        print('maxpool', x.shape)

        x = self.inception5a(x)
        print('5a', x.shape)

        x = self.inception5b(x)
        print('5b', x.shape)

        x = self.avgpool(x)
        print('AvgPool', x.shape)

        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x


# def testInceptionv1():
x = torch.randn((32, 3, 224, 224))
model = Inceptionv1(3, 1000)
# print(model(x).shape)
# return model
# model = testInceptionv1()


architecture = 'googlenet'
model_graph = draw_graph(model, input_size=(1, 3, 224, 224), roll=True, expand_nested=True,
                         graph_name=f'self_{architecture}', save_graph=True, filename=f'self_{architecture}')
# model_graph.visual_graph


# output
"""
conv1 torch.Size([1, 64, 112, 112])
maxpool1 torch.Size([1, 64, 56, 56])
conv2 torch.Size([1, 192, 56, 56])
maxpool2 torch.Size([1, 192, 28, 28])
3a torch.Size([1, 256, 28, 28])
3b torch.Size([1, 480, 28, 28])
3bmax torch.Size([1, 480, 14, 14])
4a torch.Size([1, 512, 14, 14])
4b torch.Size([1, 512, 14, 14])
4c torch.Size([1, 512, 14, 14])
4d torch.Size([1, 528, 14, 14])
4e torch.Size([1, 832, 14, 14])
maxpool torch.Size([1, 832, 7, 7])
5a torch.Size([1, 832, 7, 7])
5b torch.Size([1, 1024, 7, 7])
AvgPool torch.Size([1, 1024, 1, 1])
"""
