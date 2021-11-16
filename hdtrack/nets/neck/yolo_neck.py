import torch
import torch.nn as nn

class CBL(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1):
        super(CBL,self).__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu= nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#---------------------------------------------------#
#   SPP结构
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self,pool_sizes=None):
        self.pool_sizes =[5, 9, 13] if pool_sizes==None else pool_sizes
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in self.pool_sizes])
        # 特征图不变

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        # 输出维度等于x的维度*4
        return features

#---------------------------------------------------#
#   1*1的卷积 + 上采样
#---------------------------------------------------#
class MyInterpolate(nn.Module):
    def __init__(self):
        super(MyInterpolate, self).__init__()
    def forward(self,x):
        b,c,h,w=x.size()
        target_h=2*h
        target_w=2*w

        return x.view(b,c,h,1,w,1)\
            .expand(b,c,h,target_h//h,w,target_w//w)\
            .contiguous()\
            .view(b,c,target_h,target_w)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            CBL(in_channels, out_channels, 1),
            MyInterpolate()#nn.Upsample(scale_factor=2, mode='nearest')#
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(in_filters,filters_list):
    m = nn.Sequential(
        CBL(in_filters, filters_list[0], 1),
        CBL(filters_list[0], filters_list[1], 3),
        CBL(filters_list[1], filters_list[0], 1),
    )
    return m  # 输出维度为filters_list[0]

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(in_filters,filters_list):
    m = nn.Sequential(
        CBL(in_filters, filters_list[0], 1),
        CBL(filters_list[0], filters_list[1], 3),
        CBL(filters_list[1], filters_list[0], 1),
        CBL(filters_list[0], filters_list[1], 3),
        CBL(filters_list[1], filters_list[0], 1),
    )
    return m  # 输出维度为filters_list[0]

#---------------------------------------------------#
#   YOLOv4输出
#---------------------------------------------------#

class Yolo_Neck(nn.Module):
    def __init__(self):
        super(Yolo_Neck,self).__init__()
        self.conv1 = make_three_conv(1024, [512, 1024])
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv(2048, [512, 1024])

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = CBL(512, 256, 1)
        self.make_five_conv1 = make_five_conv(512, [256, 512])

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = CBL(256, 128, 1)
        self.make_five_conv2 = make_five_conv(256, [128, 256])

        self.down_sample1 = CBL(128, 256, 3, stride=2)
        self.make_five_conv3 = make_five_conv(512, [256, 512])

        self.down_sample2 = CBL(256, 512, 3, stride=2)
        self.make_five_conv4 = make_five_conv(1024, [512, 1024])


    def forward(self, x2,x1,x0):

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.conv2(P5)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.upsample1(P5)
        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(x1)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4, P5_upsample], dim=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.upsample2(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(x2)
        # 52,52,128 + 52,52,128 -> 52,52,256
        P3 = torch.cat([P3, P4_upsample], dim=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.make_five_conv2(P3)

        # 52,52,128 -> 26,26,256
        P3_downsample = self.down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample, P4], dim=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv3(P4)

        # 26,26,256 -> 13,13,512
        P4_downsample = self.down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample, P5], dim=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.make_five_conv4(P5)

        return P3,P4,P5
        # 返回的特征图size从小到大 tuple(3*tensor(B,C,H,W))
