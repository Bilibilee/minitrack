import glob
import random
import os
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np


def bbox_iou(boxes,clusters):
    # 以iou刻画box之间的距离
    # boxes的shape为(n,2)，clusters的shape为(k,2)
    # 计算iou时，中心点都一致

    interw = np.minimum(clusters[None,:,0],boxes[:,None,0])
    # clusters[None,:]为(1,k,2)，boxes[:,None]为(n,1,2)
    # 利用广播机制

    interh = np.minimum(clusters[None,:,1],boxes[:,None,1])

    inters = interw * interh # (n,k)
    area1 = boxes[:,0:1] * boxes[:,1:2] # (n,1)

    area2 = (clusters[:,0:1] * clusters[:,1:2]).T # (1,k)
    iou = inters / (area1 + area2 -inters) # (n,k) 表示第i个box与第j个聚类的iou

    return iou

def get_avg_iou(boxes,clusters,idx):
    # 检验聚类效果
    # 计算每个类别与该类别所属簇中心的iou。并且求mean均值

    # clusters[idx]为对应的最大iou的簇中心点
    interw = np.minimum(boxes[:,0:1],clusters[idx,0:1])
    interh = np.minimum(boxes[:,1:2],clusters[idx,1:2])
    inters = interw * interh  # (n,1)
    area1 = boxes[:, 0:1] * boxes[:, 1:2]  # (n,1)

    area2 = (clusters[idx, 0:1] * clusters[idx, 1:2])  # (n,1)
    iou = inters / (area1 + area2 - inters)  # (n,1) 表示第i个box与第j个聚类的iou

    return np.mean(iou)

def kmeans(boxes,k,max_iter):
    # 上一次迭代的聚类位置
    last_idx = None

    np.random.seed()

    # 随机选k个当聚类中心
    clusters = boxes[np.random.choice(len(boxes),size=k,replace = False)] # 花式索引，返回新的array

    for _ in range(max_iter):
        distance=1-bbox_iou(boxes,clusters) # 以1-iou刻画box的距离！！(n,k)

        idx = np.argmin(distance,axis=1) # 得到所属的簇,(n,)

        if (last_idx == idx).all(): # 已经收敛，不再训练
            break
        last_idx = idx

        # 求每一个类的中位点
        for j in range(k):
            clusters[j] = np.mean(boxes[idx == j],axis=0)

    return clusters,idx
'''
# VOC数据集
def load_boxes_voc(path):
    data = []
    # 对于每一个xml都寻找box
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height<=0 or width<=0:
            continue
        
        # 对于每一个目标都获得它的宽高
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width # 0~1
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height # 0~1
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)
'''

def load_boxes_helmet(xmlfilepath):
    # 注意w和h都要归一化到0~1
    # 返回 w、h
    # 该数据集的图片size为1920*1080
    temp_xml = os.listdir(xmlfilepath)
    boxes = []
    for xml in temp_xml:
        if xml.endswith(".csv"):
            xmlpath = os.path.join(xmlfilepath, xml)
            print(xmlpath)
            df = pd.read_csv(xmlpath)
            for i in range(len(df)):
                boxes.append([df['w'][i]/1920,df['h'][i]/1080])

    boxes=np.array(boxes).astype(np.float)
    np.save('./boxes.npy',boxes) #保存一下


if __name__ == '__main__':

    model_input_size = 416 # 模型输入图片大小
    anchors_num = 9
    anno_path = r'D:\Dataset\HELMET_DATASET\annotation'
    anchors_path='./yolo_anchors.txt'  # 保存的anchors文件

    # 载入所有的xml
    # 存储格式为转化为比例后的width,height
    if not os.path.exists('./boxes.npy'):
        load_boxes_helmet(anno_path)
    boxes=np.load('./boxes.npy')

    plt.scatter(boxes[:,0],boxes[:,1])  # 先观察一下box的分布
    plt.show()
    # 使用k聚类算法
    clusters,idx = kmeans(boxes,anchors_num,max_iter=500000)
    plt.scatter(boxes[:, 0], boxes[:, 1], c=idx, cmap='rainbow')  # 数据集根据索引idx来区分颜色
    plt.scatter(clusters[:, 0], clusters[:, 1],s=200, marker='x', c=list(range(anchors_num)))  # 聚类中心
    plt.show()

    clusters = clusters[np.argsort(-clusters[:,0]*clusters[:,1])]
    # 按面积倒序排序，因为我这里yolobody的特征图size是从小到大，所以anchor的size应该从大到小 ！！！

    print('acc:{:.2f}%'.format(get_avg_iou(boxes,clusters,idx) * 100)) # 评价kmeans的好坏，1最好

    anchors = clusters * model_input_size # cluster是0~1的，要映射到size上
    print('anchors:',anchors)

    # 写入文件
    f = open(anchors_path, 'w')
    for i in range(len(anchors)):
        if i == 0:
            w_h = "%d,%d" % (anchors[i][0], anchors[i][1])
        else:
            w_h = ", %d,%d" % (anchors[i][0], anchors[i][1])
        f.write(w_h)
    f.close()
'''
50万次的结果
[[111.69234447 176.04276695]
 [ 82.24499103 164.28413988]
 [ 59.72091845 151.35542234]
 [ 49.76815537 114.43313464]
 [ 33.01068948 101.33454751]
 [ 38.45187295  75.27191822]
 [ 25.29704382  68.64570377]
 [ 18.17506772  55.13741961]
 [ 12.7555107   42.34540118]]
'''