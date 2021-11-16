import torch.nn as nn
from ..neck.yolo_neck import  CBL

def yolo_head(in_filters,filters_list):
    m = nn.Sequential(
        CBL(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

def embed_head(in_filters,embedding_dim):
    return nn.Conv2d(in_filters, embedding_dim, kernel_size=3, stride=1, padding=1)