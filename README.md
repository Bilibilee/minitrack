# minitrack
**pytorch实现多目标跟踪：JDE（Joint detection and embedding）以及SDE（Seperate detection and embedding）**

**JDE实现来自[JDE论文](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1909.12605v1.pdf#=&zoom=150)**

**SDE实现来自[Deepsort](https://github.com/ZQPei/deep_sort_pytorch)**

**均基于Yolov4目标检测器**


## introduction
- [x] 检测和JDE模型支持训练。SDE的extractor模型尚不支持训练
- [x] 支持主流的目标检测和多目标跟踪metric：FPS，mAP，MOTA，ID switch…………
- [x] 整合了检测和跟踪模型，封装了JDE和SDE的数据关联步骤，可拓展性强
- [x] 均支持pytorch转换为onnx，以及使用onnxruntime推理模型

## 环境要求
pytorch

onnxruntime-gpu

numpy

opencv

tqdm

sklearn

motmetrics

lapjv

## 目标检测
### 配置
配置文件在cfg/yolov4_cfg.json，class_names即为模型对应的类别名称

onnx_model_path指定onnx模型地址，torch_model_path指定pytorch模型地址

model_image_size为推理以及训练时的模型输入大小，要求为32的倍数
### 推理
借鉴torchvision faster rcnn，nms采用batch nms，对不同类别框同时处理
```python
from PIL import Image
import cv2
from minitrack.detection import OnnxDetection,TorchDetection

image_name=r'img\street.jpg'
image=Image.open(image_name)

detect=TorchDetection(cfg_path='cfg/yolov4_cfg.json')
result=detect.detect_one_image(image,draw=True) # 返回opencv支持的numpy图片，BGR格式，同理OnnxDetection
cv2.imshow('result',result)
```

### 训练
迁移学习，先冻结backbone，再训练。

需配置anno文件，格式为[path] [x1,y1,x2,y2,label,track_id]

具体配置见json文件,可采用马赛克数据增强，标签平滑。
```python
from PIL import Image
from minitrack.detection import TorchDetection

detect=TorchDetection(cfg_path='cfg/yolov4_cfg.json')
detect.train()
```

### 验证
```python
from PIL import Image

from minitrack.detection import OnnxDetection,TorchDetection,EvalDetection
image_name=r'img/street.jpg'
image=Image.open(image_name)

detect=TorchDetection(cfg_path='cfg/yolov4_cfg.json')
eval_detect=EvalDetection(detection=detect)
eval_detect.test_fps(image,test_interval=20) # 测试fps
eval_detect.test_map(anno_path='dataset/helmet_detect_test_anno.txt',Batchsize=4) # 测试mAP，结果保存在results
```
### 转换为onnx
```python
from PIL import Image
from minitrack.detection import TorchDetection

detect=TorchDetection(cfg_path='cfg/yolov4_cfg.json')
detect.torch2onnx(batchsize=1,save_onnx_path='logs/yolov4.onnx')
```
## 多目标跟踪SDE
借鉴JDE，对Deepsort的数据关联策略做了几项改动

1.计算外观距离矩阵时，采用指数加权平均
2.卡尔曼滤波采用multi predict并行计算，代替for循环
3.lapjv代替匈牙利算法，线性分配过程速度更快

readme未完待续

## 参考
https://github.com/Cartucho/mAP  
https://github.com/ZQPei/deep_sort_pytorch
https://github.com/bubbliiiing/yolov4-pytorch
https://github.com/Zhongdao/Towards-Realtime-MOT
