from minitrack.utils.utils import (post_process, correct_boxes,image2Modelinput,match_abnormal_and_track)
from minitrack.utils.visualization import plot_results
import json
from minitrack.utils.object import Object
import numpy as np


class BaseDetection():
    def __init__(self,track_class_name,abnormal_class_name,cfg_path,type_modelinput):
        self.type_modelinput = type_modelinput
        self.track_class_name=track_class_name
        self.abnormal_class_name=abnormal_class_name
        cfg = json.load(open(cfg_path))
        self.cfg_path=cfg_path
        self.model_image_size= cfg['model_image_size']
        self.class_names=cfg['class_names']
        self.confidence=cfg['confidence']
        self.iou=cfg['iou']
        self.is_letterbox_image=cfg['is_letterbox_image']
        self.match_iou_threshold = cfg['match_iou_threshold']
        self.initialize(cfg)

    def initialize(self,cfg):
        raise NotImplementedError

    def model_inference(self,images_data):
        raise NotImplementedError


    def get_predictions(self,images_data,origin_images_shape,origin_images=None):
        outputs = self.model_inference(images_data)
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

                result = {'track': [], 'abnormal': [],'other':[]}
                for ltrb, score, label in zip(ltrbs, scores, labels):
                    obj = Object(ltrb, 'ltrb', label, score)
                    if self.class_names[obj.label] == self.track_class_name:
                        result['track'].append(obj)
                    elif self.class_names[obj.label] == self.abnormal_class_name:
                        result['abnormal'].append(obj)
                    else:
                        result['other'].append(obj)

                if origin_images is not None:
                    result=match_abnormal_and_track(result,origin_images[i],self.match_iou_threshold)

                result=result['track']+result['abnormal']+result['other']
                results.append(result)

        return results



    def detect_one_image(self, origin_image, draw=True):
        images_data,origin_image_shape,origin_image = image2Modelinput(origin_image,self.model_image_size,self.is_letterbox_image,self.type_modelinput)
        predictions = self.get_predictions(images_data, origin_image_shape,[origin_image])
        prediction = predictions[0]

        if not draw:
            return prediction
        elif len(prediction)==0:  # 无框直接返回原图
            return origin_image
        else:
            return plot_results(origin_image,self.class_names, prediction)
