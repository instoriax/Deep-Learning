import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, num_classes, embedding_dim):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = labels.unsqueeze(2).unsqueeze(3)
        noise_label = torch.cat((noise, labels), dim=1)
        return self.main(noise_label)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, num_classes, embedding_dim):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nc + nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        labels=labels.repeat_interleave(2,dim=1)
        labels = torch.nn.functional.pad(labels, (0, 16))
        labels = labels.unsqueeze(1).unsqueeze(1)
        labels = labels.expand(img.shape)
        img_label = torch.cat((img, labels), 1)
        return self.main(img_label)