import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        # cin,cout表示维度
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
            # 如果下采样，feature变小一倍
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
            # 不变
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if (self.is_downsample):
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class Extractor(nn.Module):
    def __init__(self, embedding_dim=751, reid=True):
        super(Extractor, self).__init__()
        # 输入图像 resize到 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        #B 64 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        #B 64 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        #B 128 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        #B 256 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        #B 512 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        #B 512 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 512
        if self.reid:
            # 如果reid为true，直接返回512维，不用做全连接层
            #x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x




