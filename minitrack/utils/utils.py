import numpy as np
import cv2
import torch
from torchvision.ops import nms
import math
from .np_util import np_nms
from .np_util import np_box_iou

def match_abnormal_and_track(result,origin_image,match_iou_threshold,scale_w=0.5,scale_h=0.25):
        # nohelmet和motor匹配，根据IOU，
        if len(result['track'])==0 or len(result['abnormal'])==0:
            return result

        track=[]
        abnormal=[]
        for obj in result['track']:
            track.append(obj.ltrb)
        for obj in result['abnormal']:
            abnormal.append(obj.ltrb)

        track=np.vstack(track)
        w=track[:,2]-track[:,0]
        h=track[:,3]-track[:,1]
        track[:,0]=track[:,0]+(1-scale_w)*w/2
        track[:,2]=track[:,2]-(1-scale_w)*w/2
        track[:,3]=track[:,1]+scale_h*h

        abnormal=np.vstack(abnormal)
        ious = np_box_iou(track,abnormal)

        track_indices = np.argmax(ious, axis=0)
        max_ious = np.max(ious, axis=0)
        for indice,max_iou,abnormal_obj in zip(track_indices,max_ious,result['abnormal']):
            if max_iou < match_iou_threshold:
                continue
            x1,y1,x2,y2=abnormal_obj.ltrb.astype(np.int32)
            result['track'][indice].abnormal_class_image=origin_image[y1:y2, x1:x2, :]

        return result

def generator_anchors(num_anchors,num_features,features_shape,model_image_size,anchors_shape,device):
    anchors_return =[]
    for i in range(num_features):
        feature_h = features_shape[i][1]
        feature_w = features_shape[i][0]
        scale = (model_image_size[0] / feature_w, model_image_size[1] / feature_h)

        anchors_level = torch.zeros(size=(feature_h, feature_w, num_anchors, 4), requires_grad=False, device=device,dtype=torch.float32)  # H,W,A,4

        anchors_level[:, :, :, 2:] = anchors_shape[i]

        grid_x = torch.linspace(0, feature_w - 1, feature_w).repeat(num_anchors, feature_h, 1).permute(1, 2, 0) * scale[0]
        # repeat后为(A,H,W),经过permute为(H,W,A)
        grid_y = torch.linspace(0, feature_h - 1, feature_h).repeat(num_anchors, feature_w, 1).permute(2, 1, 0) * scale[0]
        # repeat后为(A,W,H),经过permute后为(H,W,A)
        anchors_level[:, :, :, 0] = grid_x
        anchors_level[:, :, :, 1] = grid_y

        anchors_return.append(anchors_level.reshape(-1, 4))  # (H*W*A,4)

    anchors_return = torch.cat(anchors_return, dim=0)  # tensor(levels*H*W*A,4)

    return anchors_return  # tensor(levels*H*W*A,4)

def Ndarray2Modelinput(origin_image,model_image_size,is_letterbox_image,type_modelinput):
    oh, ow, _ = origin_image.shape
    iw, ih = model_image_size
    scale = min(iw / ow, ih / oh)
    nw = int(ow * scale)
    nh = int(oh * scale)
    if is_letterbox_image:
        # 给图像增加灰条，实现不失真的resize，或者直接resize
        image = cv2.resize(origin_image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        images_data = np.zeros((ih, iw, 3))
        rbg_color = (128, 128, 128)
        images_data[:, :, 0], images_data[:, :, 1], images_data[:, :, 2] = rbg_color[0], rbg_color[1], rbg_color[2]

        half_y = (ih - nh) / 2
        y_begin = math.floor(half_y)
        y_end = -math.ceil(half_y) if half_y!=0 else None

        half_x = (iw - nw) / 2
        x_begin=math.floor(half_x)
        x_end = -math.ceil(half_x) if half_x!=0 else None

        images_data[y_begin:y_end, x_begin:x_end, :] = image
    else:
        images_data = cv2.resize(origin_image, model_image_size, interpolation=cv2.INTER_CUBIC)

    images_data = np.array(images_data, dtype=np.float32) / 255.0
    images_data = np.transpose(images_data, (2, 0, 1))
    images_data=np.array([images_data])
    if type_modelinput=='tensor':
        images_data = torch.from_numpy(images_data)
    return images_data,((ow,oh),)

def image2Modelinput(origin_image,model_image_size,is_letterbox_image,type_modelinput='tensor'):
    if type_modelinput!='tensor' and type_modelinput!='ndarray':
        raise ValueError(" type_modelinput must be 'tensor' or 'ndarray' ")
    if  isinstance(origin_image,np.ndarray):
        images_data, origin_image_size = Ndarray2Modelinput(origin_image, model_image_size, is_letterbox_image,type_modelinput)
    else:
        origin_image=np.array(origin_image)
        images_data, origin_image_size = Ndarray2Modelinput(origin_image, model_image_size, is_letterbox_image,type_modelinput)
    # 确保RGB!!!
    return images_data,origin_image_size,origin_image

def correct_boxes(ltrb, model_input_shape, origin_image_shape):

    model_input_shape = np.array(model_input_shape,dtype=np.float32)
    origin_image_shape = np.array(origin_image_shape,dtype=np.float32)
    new_shape = origin_image_shape * np.min(model_input_shape / origin_image_shape)  # w、h

    offset = (model_input_shape - new_shape) / 2. / model_input_shape
    scale = model_input_shape / new_shape

    box_xy = np.concatenate([(ltrb[:, 0:1] + ltrb[:, 2:3]) / 2, (ltrb[:, 1:2] + ltrb[:, 3:4]) / 2],axis=-1) / model_input_shape
    box_wh = np.concatenate([ltrb[:, 2:3] - ltrb[:, 0:1], ltrb[:, 3:4] - ltrb[:, 1:2]], axis=-1) / model_input_shape

    box_xy = (box_xy - offset) * scale
    box_wh *= scale

    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    new_ltrb = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)

    new_ltrb *= np.concatenate([origin_image_shape, origin_image_shape], axis=0)
    new_ltrb[:, [0, 2]] = np.clip(new_ltrb[:, [0, 2]], a_min=0,a_max=origin_image_shape[0])
    new_ltrb[:, [1, 3]] = np.clip(new_ltrb[:, [1, 3]], a_min=0,a_max=origin_image_shape[1])
    return new_ltrb

def torch_box_iou(_box_a, _box_b,x1y1x2y2=False):
    '''就是算iou矩阵'''

    if x1y1x2y2==False:  # x1y1x2y2转换为xywh格式
        box_a_lt = _box_a[:,:2]-_box_a[:,2:]/2
        box_a_rb = _box_a[:,:2]+_box_a[:,2:]/2
        box_b_lt = _box_b[:,:2]-_box_b[:,2:]/2
        box_b_rb = _box_b[:, :2] + _box_b[:, 2:] / 2

        box_a = torch.cat([box_a_lt, box_a_rb], dim=1)
        box_b = torch.cat([box_b_lt, box_b_rb], dim=1)
    else:
        box_a = _box_a
        box_b = _box_b

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 求IOU
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def post_process(predictions, conf_thres=0.5, nms_thres=0.4):
    bbox,det_conf,cls_conf=predictions
    # 先转换为左上角右下角格式，因为前向传播，inplace操作没有关系
    bbox[...,0],bbox[...,2] = bbox[...,0]-bbox[...,2]/2 , bbox[...,0]+bbox[...,2]/2
    bbox[...,1],bbox[...,3] = bbox[...,1]-bbox[...,3]/2 , bbox[...,1]+bbox[...,3]/2

    output = []
    if isinstance(bbox,np.ndarray):
        for per_bbox,per_det_conf,per_cls_conf in zip(bbox,det_conf,cls_conf):
            class_label = np.argmax(per_cls_conf, axis=-1)[:,None]
            class_conf = np.max(per_cls_conf, axis=-1,keepdims=True)

            score = class_conf * per_det_conf
            conf_mask = (score >= conf_thres).squeeze()
            # 以目标置信度*类别概率，先进行第一步筛选
            per_bbox = per_bbox[conf_mask]
            score = score[conf_mask]
            class_label = class_label[conf_mask]
            if len(per_bbox) == 0:
                output.append(None)
                continue
            max_coordinate = per_bbox.max()
            offsets = class_label * (max_coordinate + 1)
            boxes_for_nms = per_bbox + offsets  # 变化bbox的坐标，对bbox进行平移
            keep = np_nms(boxes_for_nms,score[:,0] , nms_thres)
            per_bbox,score,class_label=per_bbox[keep],score[keep][:,0],class_label[keep][:,0]
            output.append([per_bbox,score,class_label])
    else:
        for per_bbox, per_det_conf, per_cls_conf in zip(bbox, det_conf, cls_conf):
            class_conf, class_label = torch.max(per_cls_conf, dim=-1, keepdim=True)
            score=per_det_conf * class_conf
            conf_mask = (score>= conf_thres).squeeze()

            # 以目标置信度*类别概率，先进行第一步筛选
            per_bbox = per_bbox[conf_mask]
            score = score[conf_mask]
            class_label = class_label[conf_mask]

            if len(per_bbox)==0:
                output.append(None)
                continue

            max_coordinate = per_bbox.max()
            offsets = class_label * (max_coordinate + 1)
            boxes_for_nms = per_bbox + offsets  # 变化bbox的坐标，对bbox进行平移
            keep = nms(boxes_for_nms, score[:,0], nms_thres)
            # 用官方torchvision实现的nms，更快，返回的下标索引按降序排序！！
            per_bbox,score,class_label=per_bbox[keep], score[keep][:,0], class_label[keep][:,0]
            output.append([per_bbox.cpu().numpy(),score.cpu().numpy(),class_label.cpu().numpy()])
    return output


def post_process_embed(predictions, embed_mask,conf_thres=0.5, nms_thres=0.4):
    bbox,det_conf,cls_conf,embed=predictions
    # 先转换为左上角右下角格式，因为前向传播，inplace操作没有关系
    bbox[...,0],bbox[...,2] = bbox[...,0]-bbox[...,2]/2 , bbox[...,0]+bbox[...,2]/2
    bbox[...,1],bbox[...,3] = bbox[...,1]-bbox[...,3]/2 , bbox[...,1]+bbox[...,3]/2

    output = []
    if isinstance(bbox,np.ndarray):
        for per_bbox,per_det_conf,per_cls_conf,per_embed in zip(bbox,det_conf,cls_conf,embed):
            class_label = np.argmax(per_cls_conf, axis=-1)[:,None]
            class_conf = np.max(per_cls_conf, axis=-1,keepdims=True)

            score = class_conf * per_det_conf
            conf_mask = (score >= conf_thres).squeeze()
            # 以目标置信度*类别概率，先进行第一步筛选
            per_bbox = per_bbox[conf_mask]
            score = score[conf_mask]
            class_label = class_label[conf_mask]
            embed_mask = embed_mask[conf_mask]
            if len(per_bbox) == 0:
                output.append(None)
                continue

            max_coordinate = per_bbox.max()
            offsets = class_label * (max_coordinate + 1)
            boxes_for_nms = per_bbox + offsets  # 变化bbox的坐标，对bbox进行平移
            keep = np_nms(boxes_for_nms,score[:,0] , nms_thres)
            per_bbox,score,class_label,per_embed=per_bbox[keep],score[keep,0],class_label[keep,0],per_embed[embed_mask[keep]]
            output.append([per_bbox,score,class_label,per_embed])
    else:
        for per_bbox, per_det_conf, per_cls_conf,per_embed in zip(bbox, det_conf, cls_conf,embed):
            class_conf, class_label = torch.max(per_cls_conf, dim=-1, keepdim=True)
            # keepdim=True，所以返回的class_conf和class_pred都是二维的:(num_bboxes,1)
            # class_conf是max的value值，表示类别概率
            # class_pred是max的下标索引，表示类别标签，从0开始
            score=per_det_conf * class_conf
            conf_mask = (score>= conf_thres).squeeze()
            embed_mask = embed_mask[conf_mask]
            # 以目标置信度*类别概率，先进行第一步筛选
            per_bbox = per_bbox[conf_mask]
            score = score[conf_mask]
            class_label = class_label[conf_mask]

            if len(per_bbox)==0:
                output.append(None)
                continue

            max_coordinate = per_bbox.max()
            offsets = class_label * (max_coordinate + 1)
            boxes_for_nms = per_bbox + offsets  # 变化bbox的坐标，对bbox进行平移
            keep = nms(boxes_for_nms, score[:,0], nms_thres)
            # 用官方torchvision实现的nms，更快，返回的下标索引按降序排序！！
            per_bbox,score,class_label,per_embed=per_bbox[keep], score[keep,0], class_label[keep,0],per_embed[embed_mask[keep]]
            output.append([per_bbox.cpu().numpy(),score.cpu().numpy(),class_label.cpu().numpy(),per_embed.cpu().numpy()])
    return output

def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue
                
            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue
                
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1]) # label
            merge_bbox.append(tmp_box)
    return merge_bbox
