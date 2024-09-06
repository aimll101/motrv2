from torch import nn
import torch
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 将特征图降维到 (1, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 应用全局平均池化，得到 (1, 256, 1, 1)
        pool = self.avg_pool(x)
        # 应用卷积层，仍然保持 (1, 256, 1, 1)
        conv = self.conv(pool)
        # 为了使批归一化能够工作，需要上采样到至少 (1, 256, H, W)，其中 H 和 W > 1
        # 这里我们使用原始输入的高度和宽度进行上采样
        h, w = x.size(2), x.size(3)
        conv = F.interpolate(conv, size=(h, w), mode='bilinear', align_corners=False)
        # 应用批量归一化和ReLU激活函数
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        # 1x1卷积
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 为不同的dilation rate创建模块
        self.convs = nn.ModuleList([
            ASPPConv(in_channels, out_channels, rate) for rate in atrous_rates
        ])
        # 全局平均池化模块
        self.global_pool = ASPPPooling(in_channels, out_channels)
        # 将全局平均池化模块的结果与1x1卷积的结果拼接
        self.concat_project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # 存储所有分支的输出
        res = []
        # 1x1卷积分支
        res.append(self.b0(x))
        # 依次执行其他卷积分支
        for conv in self.convs:
            res.append(conv(x))
        # 执行全局平均池化分支
        res.append(self.global_pool(x))
        # 沿着通道维度拼接所有分支的输出
        res = torch.cat(res, dim=1)
        # 应用最后的卷积和激活函数
        x = self.concat_project(res)
        return x

