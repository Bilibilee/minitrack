import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
'''
对数据集进行数据分析
统计图片的w和h
统计图片的长宽比


分别统计各个类别的gt的长宽比
分别统计各个类别的gt的w和h（缩放到0~1）

统计类别数量
'''
dataser_path=r'D:/Dataset/x_ray/train'
save_path= r'./'


images_wh=[]
images_ratio=[]

classes_num={}
classes_gt_wh={}
classes_gt_ratio={}

for xml_file in glob.glob('{}/*/XML/*.xml'.format(dataser_path)):
    tree = ET.parse(xml_file)
    print(xml_file)

    img_h = int(tree.findtext('./size/height'))
    img_w = int(tree.findtext('./size/width'))
    images_wh.append([img_w,img_h])
    images_ratio.append(img_w/img_h)

    root = tree.getroot()


    for obj in root.iter('object'):

        classname = obj.find('name').text
        classes_num[classname]=classes_num.get(classname,0)+1

        xmlbox = obj.find('bndbox')
        gt_w = int(float(xmlbox.find('xmax').text))-int(float(xmlbox.find('xmin').text))
        gt_h = int(float(xmlbox.find('ymax').text))-int(float(xmlbox.find('ymin').text))
        if classname in classes_gt_wh:
            classes_gt_wh[classname].append([gt_w/img_w,gt_h/img_h])
            classes_gt_ratio[classname].append(gt_w/gt_h)
        else:
            classes_gt_wh[classname]=[[gt_w/img_w,gt_h/img_h]]
            classes_gt_ratio[classname]=[gt_w / gt_h]


# 画图片长宽比 直方图
fig = plt.gcf()
plt.hist(images_ratio)
fig.savefig(save_path+'images_ratio.png')
plt.cla()

# 画图片长宽 散点图图
fig=plt.gcf()
images_wh=np.array(images_wh)
plt.scatter(images_wh[:,0],images_wh[:,1])
fig.savefig(save_path+'images_wh.png')
plt.cla()

# 画类别数量 柱状图
fig=plt.gcf()
classes_num_items=sorted( classes_num.items(), key = lambda kv:(kv[1], kv[0]) ) # 按gt数量排序
classes_label=[item[0] for item in classes_num_items]
classes_num_list=[item[1] for item in classes_num_items]
plt.barh(range(len(classes_label)), classes_num_list,tick_label=classes_label)
fig.savefig(save_path+'classes_num.png')
plt.cla()


total_classes_ratio=[]
total_classes_wh=[]
# 画类别长宽比 直方图 和 长宽二维直方图
for key in classes_gt_wh:
    fig = plt.gcf()
    total_classes_ratio.extend(classes_gt_ratio[key])
    plt.hist(classes_gt_ratio[key])
    fig.savefig(save_path+'classes_ratio/'+key+'_ratio.png')
    plt.cla()

    fig=plt.gcf()
    total_classes_wh.extend(classes_gt_wh[key])
    per_classes_gt_wh=np.array(classes_gt_wh[key])
    plt.scatter(per_classes_gt_wh[:,0],per_classes_gt_wh[:,1])
    fig.savefig(save_path+'classes_wh/'+key+'_wh.png')
    plt.cla()


# 画图片长宽比 直方图
fig = plt.gcf()
plt.hist(total_classes_ratio)
fig.savefig(save_path+'total_classes_ratio.png')
plt.cla()

# 画图片长宽 散点图图
fig=plt.gcf()
total_classes_wh=np.array(total_classes_wh)
plt.scatter(total_classes_wh[:,0],total_classes_wh[:,1])
fig.savefig(save_path+'total_classes_wh.png')
plt.cla()







