import torch.nn as nn
import torch
from minitrack.nets.backbone.CSPdarknet import CSPDarkNet
from minitrack.nets.neck.yolo_neck import Yolo_Neck
from minitrack.nets.head.head import yolo_head,embed_head
from minitrack.utils.utils import generator_anchors
from torch.jit.annotations import List,Tuple

def head_process(features,model_image_size,num_anchors,anchors,embedding_dim):
    # type: (List[torch.Tensor],Tuple[int,int],int,torch.Tensor,int) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]

    detections=torch.jit.annotate(List[torch.Tensor],[])
    embeddings=torch.jit.annotate(List[torch.Tensor],[])
    for feature in features:
        detection,embedding=feature[:,0:-embedding_dim,:,:],feature[:,-embedding_dim:,:,:]
        batch, C, H, W = detection.shape
        new_detection =detection.view(batch, num_anchors, -1, H, W).permute(0, 3, 4, 1, 2).reshape(batch, -1,C // num_anchors)
        # 经过view:(B,A,C//A,H,W),permute后:(B,H,W,A,C//A),reshape后:(B,H*W*A,C//A)
        # 注意要经过permute使得为(H,W,A)的顺序，因为生成的anchor也是这个顺序，一一对应
        # reshape会自动进行contiguous操作
        x = torch.sigmoid(new_detection[..., 0:1]) * model_image_size[0] / W  # 要缩放到原图上
        y = torch.sigmoid(new_detection[..., 1:2]) * model_image_size[1] / H

        detections.append(torch.cat([x,y,new_detection[...,2:]],dim=-1))

        new_embedding = embedding.permute(0, 2, 3, 1).reshape(batch, -1, embedding_dim)  # (B,H*W,512)
        embeddings.append(new_embedding)

    embeddings = torch.cat(embeddings, dim=1)  # (B,levels*H*W,512)

    detections= torch.cat(detections, dim=1)  # (B,levels*H*W*A,C//A)
    # 获得置信度，是否有物体

    det_conf = torch.sigmoid(detections[..., 4:5])
    cls_conf = torch.softmax(detections[..., 5:], dim=-1)  # (B,levels*H*W*A,-1)改成了softmax
    # decode
    x = detections[..., 0:1] + anchors[..., 0:1] # x
    y = detections[..., 1:2] + anchors[..., 1:2] # y
    w = torch.exp(detections[..., 2:3]) * anchors[..., 2:3] # w
    h = torch.exp(detections[..., 3:4]) * anchors[..., 3:4] # h
    xywh=torch.cat([x,y,w,h],dim=-1)

    return xywh,det_conf,cls_conf,embeddings


class JDE_Net(nn.Module):
    def __init__(self, num_anchors, num_classes,embedding_dim):
        super(JDE_Net, self).__init__()
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
        self.embed_head3 = embed_head(128, embedding_dim)
        # num_anchors*(5+num_classes)
        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head(256 ,[512, final_out_filter1])
        self.embed_head2 = embed_head(256, embedding_dim)

        # num_anchors*(5+num_classes)
        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head(512 ,[1024, final_out_filter0])
        self.embed_head1 = embed_head(512, embedding_dim)

        for m in self.modules():
            if isinstance(m, (nn.Linear,nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight.data,a=0.1,nonlinearity='leaky_relu')

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)
        P3 ,P4 ,P5 =self.yolo_neck(x2 ,x1 ,x0)

        #   第三个特征层52,52
        detection2 = self.yolo_head3(P3)
        embedding2 = self.embed_head3(P3)
        #   第二个特征层26,26
        detection1 = self.yolo_head2(P4)
        embedding1 = self.embed_head2(P4)
        #   第一个特征层13,13
        detection0 = self.yolo_head1(P5)
        embedding0 = self.embed_head1(P5)

        result0=torch.cat([detection0,embedding0],dim=1)
        result1=torch.cat([detection1,embedding1],dim=1)
        result2=torch.cat([detection2,embedding2],dim=1)
        return [result0,result1,result2]
        # 返回的特征图size从小到大 tuple(3*tensor(B,C,H,W))


class JDE(nn.Module):
    def __init__(self ,embedding_dim,strides,class_names ,anchors_shape ,model_image_size ,cuda):
        # class_names:list[str]
        # anchors_shape:tensor(num_features,num_anchors,2)
        # model_image_size:tuple(w,h)
        super(JDE ,self).__init__()
        self.embedding_dim = embedding_dim
        self.strides = strides # 从大到小
        self.class_names =class_names
        self.anchors_shape =anchors_shape
        self.model_image_size =model_image_size
        self.features_shape = [[model_image_size[0] // stride, model_image_size[1] // stride] for stride in self.strides]

        self.num_classes =len(class_names)
        self.num_features =len(self.strides)
        self.num_anchors =len(anchors_shape[0])

        device = torch.device('cuda' if cuda else 'cpu')
        self.anchors =generator_anchors(self.num_anchors, self.num_features, self.features_shape, model_image_size, anchors_shape, device)


        self.net =JDE_Net(self.num_anchors ,self.num_classes,self.embedding_dim).to(device)

    def forward(self ,images_data):

        features =self.net(images_data)
        predictions =head_process(features ,self.model_image_size ,self.num_anchors ,self.anchors,self.embedding_dim)

        return predictions

