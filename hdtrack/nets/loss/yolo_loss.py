import torch
import torch.nn as nn
from hdtrack.utils.utils import torch_box_iou
from ..yolov4 import head_process
from hdtrack.utils.loss_util import box_ciou,LogSoftmaxLoss,BCELoss,get_smooth_labels

class YOLOLoss(nn.Module):
    def __init__(self, anchors,anchors_shape,num_classes,num_anchors,num_features,model_image_size,device,label_smooth=0, mean=False):
        super(YOLOLoss, self).__init__()

        self.num_anchors = num_anchors
        self.num_features=num_features
        self.num_classes = num_classes
        self.model_image_size = model_image_size  # w、h

        self.label_smooth = label_smooth

        self.ignore_threshold = 0.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_loc = 1.0

        self.device = device
        self.mean = mean
        self.anchors=anchors# 输出：tensor(levels*H*W*A,4)
        self.anchors_shape=anchors_shape.reshape(-1,2) #tensor(num_features,num_anchors,2)转换为tensor(num_features*num_anchors,2)

    def forward(self, input, targets=None):
        # input:tuple(3*tensor(B,C,H,W))

        self.features_size = [[feature.size(3), feature.size(2)] for feature in input]  # w、h
        batchsize = input[0].shape[0]

        targets_each_batch = [len(target) for target in targets]
        targets = torch.cat(targets, dim=0).to(self.device)  # (-1,5 or 6)

        pred_box,conf,pred_cls = head_process(input, self.model_image_size, self.num_anchors, self.anchors)
        # 转换为tensor(B,levels*H*W*A,-1),其中conf为(B,levels*H*W*A)

        obj_mask, noobj_mask = self.match_obj(targets, targets_each_batch)  # (B,levels*H*W*A)
        noobj_mask = self.match_noobj(self.anchors, targets, targets_each_batch, noobj_mask)  # (B,levels*H*W*A)
        # (B,levels*H*W*A)


        # 计算置信度的loss
        obj_labels = (obj_mask >= 0).int()
        loss_ = BCELoss( conf.squeeze() , obj_labels )
        loss_conf = torch.sum( loss_ * obj_labels ) + torch.sum( loss_ * noobj_mask )
        # *obj_mask是算目标的置信度，当为目标时，obj_mask=1，当是忽略和背景时，obj_mask=0
        # *noobj_mask是算背景的置信度，当为背景时，noobj_mask=1，当是忽略和目标时，noobj_mask=0
        # 若一个gt，则与之交并比最大的anchor是目标，没有阈值判断，只要与gt交并比最大
        # 若一个anchor，与所有gt的最大交并比都小于threshold，则被认为背景
        # 假设一个gt与最大交并比anchor的iou都小于threshold，且该anchor与所有gt的iou都小于thresold
        # 那么mask=1，noobj_mask=0，则在目标置信度时会计算
        if targets.numel() == 0:
            loss_cls = torch.tensor(0, device=self.device, requires_grad=False)
            loss_loc = torch.tensor(0, device=self.device, requires_grad=False)
        else:
            targets_bbox = targets[obj_mask[obj_mask >= 0]][:, 0:4]
            box_loss_scale = 2 - targets_bbox[:, 2] / self.model_image_size[0] * targets_bbox[:, 3] / self.model_image_size[1]
            # box_loss_scale = 2 - 相对面积，值的范围是(1~2)，边界框的尺寸越小，bbox_loss_scale 的值就越大。box_loss_scale可以弱化边界框尺寸对损失值的影响
            # (B,levels*H*W*A)

            # 计算预测结果和真实结果的CIOU
            ciou = (1 - box_ciou(pred_box[obj_mask >= 0], targets_bbox, self.device)) * box_loss_scale
            # obj_mask[obj_mask>=0] 逻辑索引，返回的是一维的非零的整数数组
            # targets[obj_mask[obj_mask>=0] 花式索引
            # box_loss_scale[obj_mask>=0]逻辑索引，返回的是一维数组，所以要求box_ciou也要一维数组
            loss_loc = torch.sum(ciou)

            cls_labels = targets[obj_mask[obj_mask >= 0]][:, 4].long()  # 一维
            onehot_cls_labels = torch.zeros((len(cls_labels), self.num_classes), requires_grad=False, device=self.device,
                                            dtype=torch.float)
            onehot_cls_labels[[i for i in range(len(cls_labels))], cls_labels] = 1  # 花式索引
            smooth_labels = get_smooth_labels(onehot_cls_labels, self.label_smooth, self.num_classes)
            loss_cls = torch.sum(LogSoftmaxLoss(pred_cls[obj_mask >= 0], smooth_labels))

        loss = loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc
        print(loss_conf,loss_cls,loss_loc)
        # print(loss_conf.item(),loss_cls.item(),loss_loc.item())
        if self.mean:
            # 是否对损失进行归一化，用于改变loss的大小
            # 用于决定计算最终loss是除上batch_size还是除上正样本数量
            num_pos = torch.sum(obj_mask >= 0)  # 目标数量
            num_pos = torch.clamp(num_pos, min=1)  # 至少为1
        else:
            num_pos = batchsize

        return loss / num_pos



    def match_obj(self, targets, targets_each_batch):
        # target:(-1,5),5表示x,y,w,h,label
        # targets_each_batch:list,记录每张图片的target数量
        # 返回obj_mask_return,non_mask_return
        # obj_mask_return(B,levels*H*W*A)若anchor为目标则为最大iou的gt的索引，否则为-1
        # non_mask_return(B,levels*H*W*A)若anchor为背景则为1；目标或忽略则为0

        bs = len(targets_each_batch)
        # levelsxHxWxA=anchors.shape[0]
        obj_mask = [
            torch.full((bs, self.features_size[i][1], self.features_size[i][0], self.num_anchors), -1, requires_grad=False,
                       device=self.device, dtype=torch.long) for i
            in range(self.num_features)]
        # list(tensor(B,H,W,A)*levels) 初始值为-1，因为要被作为花式索引下标，所以dtype=torch.long
        noobj_mask = [torch.ones((bs, self.features_size[i][1], self.features_size[i][0], self.num_anchors), requires_grad=False,
                                 device=self.device, dtype=torch.long)
                      for i in range(self.num_features)]
        # list(tensor(B,H,W,A)*levels) 初始值为1

        features_size = torch.tensor(self.features_size, requires_grad=False, device=self.device,
                                     dtype=torch.long)  # 要做花式索引，先转换为tensor
        if targets.numel()>0:
            gt = targets[:,2:4]
            gt = torch.cat([torch.zeros((len(targets), 2), device=self.device), gt], dim=1)  # 中心点为(0,0)
            anchors_shape = torch.cat([torch.zeros((len(self.anchors_shape), 2), device=self.device), self.anchors_shape], dim=1)  # 中心点为(0,0)
            # tensor(num_features*num_anchors,2)
            iou_matrix = torch_box_iou(gt, anchors_shape, x1y1x2y2=False)

            cumsum = 0
            for batch_i, target_each_batch in enumerate(targets_each_batch):
                if target_each_batch == 0:
                    continue

                _, max_indices = torch.max(iou_matrix[cumsum:cumsum + target_each_batch,:], dim=1)
                # 为每一个gt找iou最大的anchor_shape,max_indices为tensor(target_each_batch,1)
                levelxA = max_indices
                level = levelxA // self.num_anchors
                A = levelxA % self.num_anchors

                X = targets[cumsum:cumsum + target_each_batch, 0] / self.model_image_size[0] * features_size[level, 0]
                Y = targets[cumsum:cumsum + target_each_batch, 1] / self.model_image_size[1] * features_size[level, 1]
                X = torch.floor(X).int()
                Y = torch.floor(Y).int()

                i = 0
                for l, y, x, a in zip(level, Y, X, A):
                    # 遍历每一个target，为与之iou最大的anchor赋值
                    obj_mask[l.item()][batch_i, y.item(), x.item(), a.item()] = cumsum + i  # target的下标
                    i += 1
                    noobj_mask[l.item()][batch_i, y.item(), x.item(), a.item()] = 0

                cumsum += target_each_batch

        obj_mask_return = []
        noobj_mask_return = []
        for i in range(len(obj_mask)):
            obj_mask_return.append(obj_mask[i].reshape(bs, -1))
            noobj_mask_return.append(noobj_mask[i].reshape(bs, -1))
        obj_mask_return = torch.cat(obj_mask_return, dim=1)
        noobj_mask_return = torch.cat(noobj_mask_return, dim=1)

        return obj_mask_return, noobj_mask_return

    def match_noobj(self, anchors, targets, targets_each_batch, noobj_mask):
        # non_mask_return:(B,levels*H*W*A)若anchor为背景则为1；目标或忽略则为0
        # anchors:(levels*H*W*A,4)
        # target:(-1,5),5表示x,y,w,h,label
        # targets_each_batch:list,记录每张图片的target数量
        if targets.numel()==0:
            return noobj_mask

        iou_matrix = torch_box_iou(targets[:, 0:4], anchors, x1y1x2y2=False)
        # iou矩阵

        cumsum = 0
        for batch_i, target_each_batch in enumerate(targets_each_batch):
            if target_each_batch == 0:
                continue
            max_iou_val, _ = torch.max(iou_matrix[cumsum:cumsum + target_each_batch, :], dim=0)
            # 为每一个anchors找iou最大的gt,返回iou值
            noobj_mask[batch_i][max_iou_val > self.ignore_threshold] = 0  # 小于self.ignore_threshold才被认为是背景
            cumsum += target_each_batch
        return noobj_mask




