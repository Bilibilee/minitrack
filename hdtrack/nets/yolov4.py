import torch.nn as nn
import torch
from hdtrack.nets.backbone.CSPdarknet import CSPDarkNet
from hdtrack.nets.neck.yolo_neck import Yolo_Neck
from hdtrack.nets.head.head import yolo_head

from hdtrack.utils.utils import generator_anchors
from torch.jit.annotations import List,Tuple


def head_process(features,model_image_size,num_anchors,anchors):
    # type: (List[torch.Tensor],Tuple[int,int],int,torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]
    # features:tuple(3*tensor(B,C,H,W))
    # 转换为tensor(B,levels*H*W*A,5+numclasses)
    predictions=torch.jit.annotate(List[torch.Tensor],[])
    for feature in features:
        batch, C, H, W = feature.shape
        prediction =feature.view(batch, num_anchors, -1, H, W).permute(0, 3, 4, 1, 2).reshape(batch, -1,C // num_anchors)
        # 经过view:(B,A,C//A,H,W),permute后:(B,H,W,A,C//A),reshape后:(B,H*W*A,C//A)
        # 注意要经过permute使得为(H,W,A)的顺序，因为生成的anchor也是这个顺序，一一对应
        # reshape会自动进行contiguous操作
        x = torch.sigmoid(prediction[..., 0:1]) * model_image_size[0] / W  # 要缩放到原图上
        y = torch.sigmoid(prediction[..., 1:2]) * model_image_size[1] / H

        predictions.append(torch.cat([x,y,prediction[...,2:]],dim=-1))

    predictions = torch.cat(predictions, dim=1)  # (B,levels*H*W*A,C//A)
    # 获得置信度，是否有物体

    det_conf = torch.sigmoid(predictions[..., 4:5])
    cls_conf = torch.softmax(predictions[..., 5:], dim=-1)  # (B,levels*H*W*A,-1)改成了softmax
    # decode
    x = predictions[..., 0:1] + anchors[..., 0:1] # x
    y = predictions[..., 1:2] + anchors[..., 1:2] # y
    w = torch.exp(predictions[..., 2:3]) * anchors[..., 2:3] # w
    h = torch.exp(predictions[..., 3:4]) * anchors[..., 3:4] # h
    xywh=torch.cat([x,y,w,h],dim=-1)

    return xywh,det_conf,cls_conf

class Yolo_Net(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(Yolo_Net, self).__init__()
        # ---------------------------------------------------#
        #   当输入图片size为(416,416)时
        #   经过PAN,输出三个特征层:
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        # ---------------------------------------------------#
        self.backbone = CSPDarkNet()

        self.yolo_neck =Yolo_Neck()

        # num_anchors*(5+num_classes)
        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head(128 ,[256, final_out_filter2])

        # num_anchors*(5+num_classes)
        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head(256 ,[512, final_out_filter1])

        # num_anchors*(5+num_classes)
        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head(512 ,[1024, final_out_filter0])

        for m in self.modules():
            if isinstance(m, (nn.Linear,nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight.data,a=0.1,nonlinearity='leaky_relu')

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)
        P3 ,P4 ,P5 =self.yolo_neck(x2 ,x1 ,x0)

        #   第三个特征层52,52
        out2 = self.yolo_head3(P3)
        #   第二个特征层26,26)
        out1 = self.yolo_head2(P4)
        #   第一个特征层13,13
        out0 = self.yolo_head1(P5)

        return [out0, out1, out2]
        # 返回的特征图size从小到大 tuple(3*tensor(B,C,H,W))


class YOLOv4(nn.Module):
    def __init__(self ,class_names ,anchors_shape ,model_image_size ,cuda):
        # class_names:list[str]
        # anchors_shape:tensor(num_features,num_anchors,2)
        # model_image_size:tuple(w,h)
        super(YOLOv4 ,self).__init__()
        self.strides = [32, 16, 8]  # 从大到小
        self.class_names =class_names
        self.anchors_shape =anchors_shape
        self.model_image_size =model_image_size
        self.features_shape=[[model_image_size[0] // stride, model_image_size[1] // stride] for stride in self.strides]

        self.num_classes =len(class_names)
        self.num_features =len(self.strides)
        self.num_anchors =len(anchors_shape[0])

        device = torch.device('cuda' if cuda else 'cpu')

        self.anchors =generator_anchors(self.num_anchors, self.num_features, self.features_shape, model_image_size, anchors_shape, device)
        self.net =Yolo_Net(self.num_anchors ,self.num_classes).to(device)

    def forward(self ,images_data):

        features =self.net(images_data)
        predictions =head_process(features ,self.model_image_size ,self.num_anchors ,self.anchors)

        return predictions
