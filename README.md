# minitrack
**针对motor的多目标跟踪，感兴趣的目标包括：motor,cyclist,pedestrian,nohelmet,helmet。其中只对motor上的驾乘人员标注是否佩戴头盔，行人、自行车手不标注**

**数据集来自[缅甸监控数据集](https://osf.io/4pwj8/)**

**pytorch实现多目标跟踪：JDE（Joint detection and embedding）以及SDE（Seperate detection and embedding）**

**JDE实现来自[JDE论文](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1909.12605v1.pdf#=&zoom=150)**

**SDE实现来自[Deepsort](https://github.com/ZQPei/deep_sort_pytorch)**

**均基于Yolov4目标检测器**

## perform
![map](./results/map.png)

## model
* 数据集：包括coco、voc、helmet。以helmet结尾——表示检测的目标是：motor,cyclist,pedestrian,nohelmet,helmet
* 模型框架：包括onnx和pytorch模型
* 模型种类：包括Detection、JDEembed、extractor，分别对应目标检测、Joint detection embed、Deepsort的extractor提取特征模型
* 百度云盘[yolov4_weights_helmet](https://pan.baidu.com/s/1PYZtnyi4wqS_SXzKqVPqQA )
  提取码：2354


## introduction
- [x] 检测和JDE模型支持训练。SDE的extractor模型尚不支持训练
- [x] 支持主流的目标检测和多目标跟踪metric：FPS，mAP，MOTA，ID switch…………
- [x] 整合了检测和跟踪模型，封装了JDE和SDE的数据关联步骤，可拓展性强
- [x] 均支持pytorch转换为onnx，以及使用onnxruntime推理模型
- [x] 支持逆行、未戴头盔检测，并将保存3张抓拍图片到本地文件夹
- [x] 车牌识别待支持

## 环境要求
pytorch

onnxruntime-gpu

numpy

opencv

tqdm

sklearn

motmetrics

lapjv


### 配置
配置文件在cfg文件夹下，以yolov4_cfg.json为例
```json
{
  "anchors_shape": [
    [[142, 110],
      [192, 243],
      [459, 401]
    ],
    [[36, 75],
      [76, 55],
      [72, 146]
    ],
    [[12, 16],
      [19, 36],
      [40, 28]
    ]
  ], 
  /*表示anchor锚框的w、h，共9种anchors，面积大的anchor对应层次越深的特征图*/
  "onnx_model_path": "logs/yolov4_weights_helmet.onnx", //yolov4的onnx模型地址
  "torch_model_path": "logs/yolov4_weights_helmet.pth", //yolov4的pytorch模型地址
  "model_image_size": [1088, 608], //模型的输入图片大小w、h，确保为32的倍数
  "confidence": 0.4, //置信度阈值
  "match_iou_threshold": 0.1, //nohelmet和motor进行iou匹配时，若iou<该阈值，则舍弃
  "iou": 0.3, //nms阶段，iou阈值
  "cuda": true, //是否使用cuda，对Torchdetection有效
  "is_letterbox_image":true, //是否对输入图片进行信纸变换
  
  //训练模型的配置信息
  "Batch_size_freeze" : 1, //冻结阶段的batch
  "Batch_size_nofreeze" : 1, //解冻阶段的batch
  "Freeze_Epoch" : 18, //冻结阶段训练的epoch数量
  "Unfreeze_Epoch" : 30, //解冻阶段训练的epoch数量
  "Freeze_lr" : 0.001, //冻结阶段的学习率
  "Unfreeze_lr" : 0.0001, //解冻阶段的学习率
  "train_detect_anno_path" : "dataset/helmet_detect_train_anno.txt", //训练集的annotation文件
  "test_detect_anno_path" : "dataset/helmet_detect_test_anno.txt", //测试集的annotation文件
  "resume" : false, //checkpoint，是否从上一断点进行训练
  "Cosine_lr" : false, //学习率是否启用余弦衰减策略
  "mosaic" : false, //是否采用马赛克增强
  "is_random" : true, //是否采用图像增强，包括图像旋转缩放、色域变化
  "smooth_label" : 0.001, //标签平滑

  "class_names": [ //类别名称
    "motor",
    "cyclist",
    "pedestrian",
    "nohelmet",
    "helmet"
  ]
}
```
## 类型
* 目标检测类(yolov4):OnnxDetection、TorchDetection
* 嵌入学习类:OnnxJdeEmbed、TorchJdeEmbed、OnnxSdeEmbed、TorchSdeEmbed
* 跟踪类(准确来说是数据关联策略):DeepsortTracker、JdeTracker
* 验证模型类:EvalDetection、EvalEmbed、EvalTracker
* 数据集导入:YoloDataset、EmbedDataset、TrackerDataset

嵌入学习Sde包括extractor和detection，跟踪类需要提供一个嵌入学习类用来提取特征和检测
## 接口
每个模型都实现了
* detect_one_image(image:`PIL.Image`,draw=True)。返回cv2支持numpy格式图片 or list[class Object]
* get_predictions(images_data:`tensor or ndarray`,origin_images_shape:`list[tuple(w,h)]`,origin_images:`list[ndarray]`)。
支持batch>1的输入，返回list(list(class Object)*batch)
* model_inference(image_datas:`PIL.Image or ndarray`)。模型前向传播
* train()。模型训练
其他方法：
* track_one_image:跟踪类实现的方法
* test_fps、test_map、test_roc、test_mota:验证模型类实现的方法

class Object——属性： 
* 定位ltrb、ltwh、xyah：`ndarray(4,) float32`
* 类别label：`int`，表示分类类别
* 得分score：`float`，等于det_conf*cls_conf
* 特征向量feature：`ndarray(embedding_dim,) float32 or None`，嵌入学习模型返回结果才能有该属性，且经过L2归一化。否则为None。
* 跟踪序号track_id：`int or None`，跟踪类返回结果才有该属性，否则为None
* 中心点坐标序列centers：`list(tuple(x,y))`，记录track的中心点坐标，长度不超过budget
* 未戴头盔的目标图片abnormal_class_image：`ndarray(H,W,C) or None`，需要跟踪的目标motor才有该属性，cv2的numpy图片格式，记录该motor上未戴头盔人头图片


## 目标检测
两种目标检测器：from minitrack.detection import OnnxDetection,TorchDetection

一个验证目标检测类：from minitrack.detection import EvalDetetcion

### 推理
实现了两种nms:

1.pytorch版本调用from torchvision.ops import nms
2.onnx版本调用numpy版本的fast nms（见np_util.py）

都借鉴了torchvision faster rcnn，nms采用batch nms，对不同类别框同时处理

```python
# 以TorchDetection为例，OnnxDetection接口相同
from PIL import Image
import cv2
from minitrack.detection import OnnxDetection,TorchDetection

image_name=r'img\street.jpg'
image=Image.open(image_name)

detect=TorchDetection(track_class_name='motor',abnormal_class_name='nohelmet',cfg_path='cfg/yolov4_cfg.json')
# track_class_name指示需要跟踪的类别名
# abnormal_class_name指示异常的类别名，即未戴头盔
# cfg_path，配置文件位置
# 注意！track_class_name和abnormal_class_name都需与.json文件保持一致
# 为了实现抓拍未戴头盔，需要匹配跟踪的motor和异常的nohelmet，如果想关闭此功能，请赋值class_names之外的其他字符串，如空字符串
result=detect.detect_one_image(image,draw=True) 
# 输入是PIL.Image数据类型
# draw为True，返回opencv支持的numpy图片，BGR格式；为False,返回list[class Object]
# class Object定义在utils.object.py

cv2.imshow('result',result)
```

### 训练
仅TorchDetection和TorchJdeEmbed支持训练，Onnx版均不支持 ，可以把训练好的pytorch转换为onnx。

迁移学习，先冻结backbone，再训练。

需配置anno文件，格式为[image_path] [x1,y1,x2,y2,label,track_id]

，请保证label与class_names匹配

具体配置见json文件,可采用马赛克数据增强，标签平滑。
```python
from minitrack.detection import TorchDetection

detect=TorchDetection('','',cfg_path='cfg/yolov4_cfg.json')
detect.train()
```

### 验证
```python
from PIL import Image

from minitrack.detection import OnnxDetection,TorchDetection,EvalDetection
image_name=r'img/street.jpg'
image=Image.open(image_name)

detect=TorchDetection('','',cfg_path='cfg/yolov4_cfg.json')
eval_detect=EvalDetection(detection=detect) # detection指定需要验证的目标检测器
eval_detect.test_fps(image,test_interval=20) # 测试fps
eval_detect.test_map(anno_path='dataset/helmet_detect_test_anno.txt',Batchsize=4) # 测试mAP，结果保存在results
```
### 转换为onnx
若报warning可忽视
```python
from minitrack.detection import TorchDetection

detect=TorchDetection('','',cfg_path='cfg/yolov4_cfg.json')
detect.torch2onnx(batchsize=1,save_onnx_path='logs/yolov4.onnx')
```
## 嵌入学习类
其中的检测模型都是yolov4。JdeEmbed是embed和detection共享一个网络，SdeEmbed将embed和detection分开，分为为extractor模型和detection模型。

嵌入学习类：from minitrack.tracker import TorchJdeEmbed,OnnxJdeEmbed,TorchSdeEmbed,OnnxSdeEmbed
### 推理
```python
# 以TorchJdeEmbed为例，TorchSdeEmbed、OnnxJdeEmbed、OnnxSdeEmbed同理
from PIL import Image
import cv2
from minitrack.tracker import TorchJdeEmbed,OnnxJdeEmbed

image_name=r'img\street.jpg'
image=Image.open(image_name)

embed=TorchJdeEmbed(track_class_name='motor',abnormal_class_name='nohelmet',cfg_path='cfg/jde_cfg.json')
result=embed.detect_one_image(image,draw=True)
# 跟detection接口一致。需要指定track_class_name,embed只对该类别提取特征。
# embed返回的Object具有feature属性，用于track的外观特征匹配

cv2.imshow('result',result)
```

### 训练
见目标检测类的训练步骤
### 验证
```python
from PIL import Image

from minitrack.tracker import OnnxJdeEmbed,TorchJdeEmbed,EvalEmbed
image_name=r'img/street.jpg'
image=Image.open(image_name)

embed=TorchJdeEmbed('','',cfg_path='cfg/yolov4_cfg.json')
embed.test_roc(anno_path,Batchsize,save_path = './results') # 验证ROC曲线
```
### 转换为onnx
见目标检测类

## 多目标跟踪类
准确来说是数据关联策略

借鉴JDE，对Deepsort的数据关联策略做了几项改动

1.计算外观距离矩阵时，采用指数加权平均
2.卡尔曼滤波采用multi predict并行计算，代替for循环
3.lapjv代替匈牙利算法，线性分配过程速度更快

### 推理
```python
from minitrack.tracker import EvalTracker,DeepsortTracker
from minitrack.tracker import OnnxSdeEmbed,TorchSdeEmbed
from PIL import Image

embed=OnnxSdeEmbed('motor','nohelmet')
tracker=DeepsortTracker(embed)
image_name=r'img\street.jpg'
image=Image.open(image_name)
tracker.track_one_image(image,draw=True,detect_abnormal_behavior=True,save_path='./results',right_direction='y-',count_threshold=0.7,ignore_count_num=8)
# 这里只是对track_one_image参数做一个说明,输入要求是连续帧、视频
# detect_abnormal_behavior=True 是否进行逆行和未戴头盔抓拍
# save_path='./results' 抓拍后的图片存在位置
# right_direction='y-' 正确的方向['x+','x-','y+','y1']
# count_threshold=0.7 逆行次数大于该阈值，则判定为逆行
# ignore_count_num=8 若跟踪数量小于该值，则不进行逆行判定

```
### 验证
```python
from minitrack.tracker import EvalTracker,DeepsortTracker
from minitrack.tracker import OnnxSdeEmbed,TorchSdeEmbed

embed=OnnxSdeEmbed('motor','nohelmet')
tracker=DeepsortTracker(embed)
eval=EvalTracker(tracker)

eval.test_mota(anno_path='dataset/helmet_embed_test_anno.txt') #利用motmetrics库计算多目标跟踪指标
```


## 调用视频
详见video.py


## 参考
https://github.com/Cartucho/mAP  
https://github.com/ZQPei/deep_sort_pytorch
https://github.com/bubbliiiing/yolov4-pytorch
https://github.com/Zhongdao/Towards-Realtime-MOT
