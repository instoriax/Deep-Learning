# print("Please define your ResNet50 in this file.")
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, skip):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(out_channels//4)

        if skip==1 and in_channels!=64:
            self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        else:
            self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels//4)

        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.skip = skip
        if skip==1 and in_channels!=64:
            self.skipconv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
            self.skipbn = nn.BatchNorm2d(out_channels)
        elif skip==1:
            self.skipconv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            self.skipbn = nn.BatchNorm2d(out_channels)            


    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.skip==1:
            residual = self.skipconv(residual)
            residual = self.skipbn(residual)

        x += residual
        x = self.relu(x)

        return x

class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()

        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.layer1 = nn.Sequential(
            Bottleneck(64, 256, 1),
            Bottleneck(256, 256, 0),
            Bottleneck(256, 256, 0),
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 512, 1),
            Bottleneck(512, 512, 0),
            Bottleneck(512, 512, 0),
            Bottleneck(512, 512, 0),
        )
        self.layer3 = nn.Sequential(
            Bottleneck(512, 1024, 1),
            Bottleneck(1024, 1024, 0),
            Bottleneck(1024, 1024, 0),
            Bottleneck(1024, 1024, 0),
            Bottleneck(1024, 1024, 0),
            Bottleneck(1024, 1024, 0),
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 2048, 1),
            Bottleneck(2048, 2048, 0),
            Bottleneck(2048, 2048, 0),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


