import torch
import torch.nn as nn
import torch.nn.functional as F
#-------------------------------------------------#
#   MISH激活函数
#-------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#---------------------------------------------------#
#   卷积块 -> 卷积 + 批量归一化 + 激活函数
#   stride=1时特征图大小不变，stride=2时特征图大小*1/2
#---------------------------------------------------#
class CBM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CBM, self).__init__()
        self.conv =nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   残差块加SAM注意力机制
#   SAM是CBAM中的spatial attention module
#   yolov4这里做了一点改到，把pool操作改成1*1的conv，空间注意力机制变换为点注意力机制
#---------------------------------------------------#
class SAM(nn.Module):
    def __init__(self,in_planes):
        super(SAM,self).__init__()
        self.conv=nn.Conv2d(in_planes,in_planes,kernel_size=1,stride=1,padding=0)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        out=self.conv(x)
        out=self.sigmoid(out)
        return x*out

class Resblock(nn.Module):
    # 特征图大小不变，维度*2
    def __init__(self, in_channels, hidden_channels=None,used_attention=False):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        self.block = nn.Sequential(
            CBM(in_channels, hidden_channels, 1),
            CBM(hidden_channels, in_channels, 3)
        )
        if used_attention==True:
            self.attention=SAM(in_channels)
        else:
            self.attention=None

    def forward(self, x):
        out=self.block(x)
        if self.attention is not None:
            out=self.attention(out)

        return x + out

#-----------------------------------------------------------------------------#
#   由resblock残差块组成的大结构块
#                       ^----------split_conv0--------->
#                       |                              |
#   input->downsample_conv-->split_conv1-->blocks_conv-->concat_conv-->out
#------------------------------------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, is_first):
        super(Resblock_body, self).__init__()
        # stride=2,进行下采样
        self.downsample_conv = CBM(in_channels, out_channels, kernel_size=3, stride=2)

        if is_first:  # 是否是第一个大结构块
            # identity残差边
            self.split_conv0 = CBM(out_channels, out_channels, 1)

            self.split_conv1 = CBM(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels, out_channels//2) for _ in range(num_blocks)],
                CBM(out_channels, out_channels, 1)
            )
            self.concat_conv = CBM(out_channels*2, out_channels, 1)
        else:
            # identity残差边
            self.split_conv0 = CBM(out_channels, out_channels//2, 1)

            # 堆叠多个残差块
            self.split_conv1 = CBM(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                CBM(out_channels//2, out_channels//2, 1)
            )

            self.concat_conv = CBM(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        identity = self.split_conv0(x)

        out = self.split_conv1(x)
        out = self.blocks_conv(out)

        out = torch.cat([out, identity], dim=1)

        out = self.concat_conv(out)

        return out

class CSPDarkNet(nn.Module):
    def __init__(self):
        super(CSPDarkNet, self).__init__()

        layers=[1, 2, 8, 8, 4]
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = CBM(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            # 416,416,32 -> 208,208,64
            Resblock_body(self.inplanes, self.feature_channels[0], num_blocks=layers[0], is_first=True),
            # 208,208,64 -> 104,104,128
            Resblock_body(self.feature_channels[0], self.feature_channels[1], num_blocks=layers[1], is_first=False),
            # 104,104,128 -> 52,52,256
            Resblock_body(self.feature_channels[1], self.feature_channels[2], num_blocks=layers[2], is_first=False),
            # 52,52,256 -> 26,26,512
            Resblock_body(self.feature_channels[2], self.feature_channels[3], num_blocks=layers[3], is_first=False),
            # 26,26,512 -> 13,13,1024
            Resblock_body(self.feature_channels[3], self.feature_channels[4], num_blocks=layers[4], is_first=False)
        ])

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data)# 就是kaiming初始化
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5
