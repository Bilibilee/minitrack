import torch
from PIL import Image
import cv2

from minitrack.detection import OnnxDetection,TorchDetection,EvalDetection
from minitrack.tracker import EvalTracker,EvalEmbed,DeepsortTracker
from minitrack.tracker import OnnxSdeEmbed,TorchSdeEmbed,OnnxJdeEmbed,TorchJdeEmbed
image_name=r'img\street.jpg'
image=Image.open(image_name)

detect=TorchDetection(cfg_path='cfg/yolov4_cfg.json')
detect.torch2onnx(batchsize=1,save_onnx_path='logs/yolov4_cfg.json')
result=detect.detect_one_image(image,draw=True)
cv2.imshow('result',result)
eval_detect=EvalDetection(detection=detect)
eval_detect.test_fps(image,test_interval=20)
eval_detect.test_map(anno_path='dataset/helmet_detect_test_anno.txt',Batchsize=4)

embed=OnnxSdeEmbed(['motor'])
tracker=DeepsortTracker(embed)
eval=EvalTracker(tracker)

eval.test_mota(anno_path='dataset/helmet_embed_test_anno.txt')
'''
import cv2
import os
fps = 10
size = (1920, 1080)
videowriter = cv2.VideoWriter("save.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

list_jpg_names = os.listdir('./results')
#print(list_jpg_names)

for i in range(12, 149):
    name='%d.jpg' % i
    if name in list_jpg_names:
        img = cv2.imread('./results/%d.jpg' % i)
        videowriter.write(img)
'''

