from minitrack.utils.utils import (post_process, correct_boxes,image2Modelinput)
from minitrack.utils.visualization import plot_results
import json
from minitrack.utils.object import Object
import numpy as np

class BaseDetection():
    def __init__(self,cfg_path,type_modelinput):
        self.type_modelinput = type_modelinput
        cfg = json.load(open(cfg_path))
        self.cfg_path=cfg_path
        self.model_image_size= cfg['model_image_size']
        self.class_names=cfg['class_names']
        self.confidence=cfg['confidence']
        self.iou=cfg['iou']
        self.is_letterbox_image=cfg['is_letterbox_image']
        self.initialize(cfg)

    def initialize(self,cfg):
        raise NotImplementedError

    def model_inference(self,images_data):
        raise NotImplementedError

    def get_predictions(self,images_data,origin_images_shape):
        # 输入images_data是tensor(B,C,H,W),origin_image_shape:list((w,h)*B)
        # 返回list[ dict{'boxes':tensor(n,4),'labels':tensor(n),'scores':tensor(n)} * B ]
        # 注意labels为整数，且labels和scores都是一维的tensor。boxes为x1,y1,x2,y2
        # 当没有预测框时，为None
        outputs = self.model_inference(images_data)
        # tensor(B,levels*H*W*A,5+numclasses),xywh
        batch_detections = post_process(outputs, conf_thres=self.confidence, nms_thres=self.iou)
        # list[ndarray([num_anchors, 7])*B] 7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_label
        # 若无预测框，则为None
        results = []
        for i, batch_detection in enumerate(batch_detections):
            if batch_detection is None:
                results.append([])
            else:
                ltrbs, scores, labels = batch_detection
                if self.is_letterbox_image:
                    # 将box映射回origin_image
                    ltrbs = correct_boxes(ltrbs, self.model_image_size, origin_images_shape[i])
                else:
                    ltrbs[:, [0, 2]] = np.clip(ltrbs[:, [0, 2]] / self.model_image_size[0] * origin_images_shape[i][0],
                                              a_min=0, a_max=origin_images_shape[i][0])
                    ltrbs[:, [1, 3]] = np.clip(ltrbs[:, [1, 3]] / self.model_image_size[1] * origin_images_shape[i][1],
                                              a_min=0, a_max=origin_images_shape[i][1])
                result = []
                for ltrb, score, label in zip(ltrbs, scores, labels):
                    obj = Object(ltrb, 'ltrb', label, score)
                    result.append(obj)
                results.append(result)

        return results

    def detect_one_image(self, origin_image, draw=True):
        images_data,origin_image_shape,origin_image = image2Modelinput(origin_image,self.model_image_size,self.is_letterbox_image,self.type_modelinput)
        predictions = self.get_predictions(images_data, origin_image_shape)
        prediction = predictions[0]
        if not draw:
            return prediction
        elif len(prediction)==0:  # 无框直接返回原图
            return origin_image
        else:
            return plot_results(origin_image,self.class_names, prediction)