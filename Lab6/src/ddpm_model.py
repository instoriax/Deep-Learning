import torch
from torch import nn
from diffusers import UNet2DModel

class ClassConditionedUnet(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = UNet2DModel(
        sample_size=64,
        in_channels=6,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512), 
        down_block_types=(
            "DownBlock2D", 
            "DownBlock2D",
            "DownBlock2D", 
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D", 
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
          ),
    )
  def forward(self, x, t, one_hot_labels):
    labels=one_hot_labels.repeat_interleave(2,dim=1)
    labels = torch.nn.functional.pad(labels, (0, 16))
    labels = labels.unsqueeze(1).unsqueeze(1)
    labels = labels.expand(x.shape)
    net_input = torch.cat((x, labels), 1)
    return self.model(net_input, t).sample
  