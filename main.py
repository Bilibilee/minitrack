

from PIL import Image

from minitrack.detection import OnnxDetection,TorchDetection,EvalDetection
from minitrack.tracker import EvalTracker,EvalEmbed,DeepsortTracker
from minitrack.tracker import OnnxSdeEmbed,TorchSdeEmbed,OnnxJdeEmbed,TorchJdeEmbed



embed=OnnxSdeEmbed('motor','nohelmet')
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
