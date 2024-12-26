import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, skip):
        super(Bottleneck, self).__init__()

        if skip==1:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.skip = skip
        if skip==1:
            self.skipconv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
            self.skipbn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        if self.skip==1:
            residual = self.skipconv(residual)
            residual = self.skipbn(residual)

        x = x + residual
        x = self.relu(x)

        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),            
        ) 
    def forward(self, x):
        x = self.block(x)
        return x

class ResNet34_UNet(nn.Module):

    def __init__(self):
        super(ResNet34_UNet, self).__init__()

        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.encoder1 = nn.Sequential(
            Bottleneck(64, 64, 0),
            Bottleneck(64, 64, 0),
            Bottleneck(64, 64, 0),
        )
        self.encoder2 = nn.Sequential(
            Bottleneck(64, 128, 1),
            Bottleneck(128, 128, 0),
            Bottleneck(128, 128, 0),
            Bottleneck(128, 128, 0),
        )
        self.encoder3 = nn.Sequential(
            Bottleneck(128, 256, 1),
            Bottleneck(256, 256, 0),
            Bottleneck(256, 256, 0),
            Bottleneck(256, 256, 0),
            Bottleneck(256, 256, 0),
            Bottleneck(256, 256, 0),
        )
        self.encoder4 = nn.Sequential(
            Bottleneck(256, 512, 1),
            Bottleneck(512, 512, 0),
            Bottleneck(512, 512, 0),
        )

        self.bottleneck = Block(512,256)

        self.upconv4 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.decoder4 = Block(384, 32)
        self.upconv3 = nn.ConvTranspose2d(288, 144, kernel_size=2, stride=2)
        self.decoder3 = Block(144, 32)
        self.upconv2 = nn.ConvTranspose2d(160, 80, kernel_size=2, stride=2)
        self.decoder2 = Block(80, 32)
        self.upconv1 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.decoder1 = Block(48, 32)

        self.outconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            Block(32, 32),
            Block(32, 1),
        )

        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)

        bottleneck = self.bottleneck(encoder4)

        decoder4 = torch.cat((encoder4, bottleneck), dim=1)
        decoder4 = self.upconv4(decoder4)
        decoder4 = self.decoder4(decoder4)

        decoder3 = torch.cat((encoder3, decoder4), dim=1)
        decoder3 = self.upconv3(decoder3)
        decoder3 = self.decoder3(decoder3)

        decoder2 = torch.cat((encoder2, decoder3), dim=1)
        decoder2 = self.upconv2(decoder2)
        decoder2 = self.decoder2(decoder2)

        decoder1 = torch.cat((encoder1, decoder2), dim=1)
        decoder1 = self.upconv1(decoder1)
        decoder1 = self.decoder1(decoder1)

        x = self.outconv(decoder1)
        x = self.bn(x)
        x = self.sigmoid(x)
        return x
    


