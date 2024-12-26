import torch
import torch.nn as nn

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

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.encoder1 = Block(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = Block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = Block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = Block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = Block(128, 64)

        self.outconv = nn.Conv2d(64, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(self.pool1(encoder1))
        encoder3 = self.encoder3(self.pool2(encoder2))
        encoder4 = self.encoder4(self.pool3(encoder3))

        bottleneck = self.bottleneck(self.pool4(encoder4))

        decoder4 = self.upconv4(bottleneck)
        decoder4 = torch.cat((encoder4, decoder4), dim=1)
        decoder4 = self.decoder4(decoder4)
        decoder3 = self.upconv3(decoder4)
        decoder3 = torch.cat((encoder3, decoder3), dim=1)
        decoder3 = self.decoder3(decoder3)
        decoder2 = self.upconv2(decoder3)
        decoder2 = torch.cat((encoder2, decoder2), dim=1)
        decoder2 = self.decoder2(decoder2)
        decoder1 = self.upconv1(decoder2)
        decoder1 = torch.cat((encoder1, decoder1), dim=1)
        decoder1 = self.decoder1(decoder1)

        x = self.outconv(decoder1)
        x = self.bn(x)
        x = self.sigmoid(x)

        return x


