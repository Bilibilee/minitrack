import torch
import numpy as np

def np_nms(ltrb,scores,nms_threshold=0.4):
    # Sort boxes
    sorted_index = np.argsort(-scores)
    #sorted_class_labels = class_labels[sorted_index]
    sorted_ltrb = ltrb[sorted_index]

    ious = np_box_iou(sorted_ltrb,sorted_ltrb)

    ious = np.triu(ious, k=1) #上三角矩阵，对角线为0

    keep = ious.max(axis=0) # 每一列最大值
    keep = keep < nms_threshold
    return sorted_index[keep]


def np_box_iou( boxes1, boxes2):
    """
    boxes1:[N, 4]
    boxes2:[M, 4]
    ious:[N, M]
    """
    area1 = (boxes1.T[2] - boxes1.T[0]) * (boxes1.T[3] -boxes1.T[1])
    area2 = (boxes2.T[2] - boxes2.T[0]) * (boxes2.T[3] -boxes2.T[1])

    # boxes1[:, None, :2] shape:[4125, 1, 2], boxes2[:, :2] shape:[4125, 2]
    overlap_area_left_top = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    overlap_area_right_bot = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    overlap_area_sizes = np.clip(overlap_area_right_bot -overlap_area_left_top,a_min=0,a_max=None)
    overlap_area = overlap_area_sizes[:, :, 0] * overlap_area_sizes[:, :,1]
    ious = overlap_area / (area1[:, None] + area2 - overlap_area)

    return ious


if __name__=='__main__':
    # 百个框差别不大，千个框要0.3秒，差了十倍
    import time
    from torchvision.ops import nms
    ltrb=np.random.rand(200,4)
    scores=np.random.rand(200,1)[:,0]
    t1=time.time()
    keep1=np_nms(ltrb,scores,0.4)
    t2=time.time()
    print(t2-t1,keep1)

    ltrb=torch.from_numpy(ltrb)
    scores=torch.from_numpy(scores)
    t3=time.time()
    keep2=nms(ltrb, scores, 0.4)
    t4=time.time()
    print(t4-t3,keep2)