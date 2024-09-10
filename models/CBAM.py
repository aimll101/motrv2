from torch import nn
import torch
from ChannelAttention import ChannelAttention
from SpatialAttention import SpatialAttention


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


