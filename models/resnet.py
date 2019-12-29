import torch
import torch.nn as nn
from torchvision.models import BasicBlock

class Resnet18Features(nn.Module):
    def __init__(self, in_channels=5, norm_layer=nn.BatchNorm2d):
        super(Resnet18Features, self).__init__()
        
        self.in_channels = in_channels
        self.norm_layer = norm_layer

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = BasicBlock(64, 64, norm_layer=self.norm_layer)
        self.layer2 = BasicBlock(64, 128, stride=2, norm_layer=self.norm_layer)
        self.layer3 = BasicBlock(128, 256, stride=2, norm_layer=self.norm_layer)
        self.layer4 = BasicBlock(256, 512, stride=2, norm_layer=self.norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x