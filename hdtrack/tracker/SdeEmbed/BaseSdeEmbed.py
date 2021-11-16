from hdtrack.utils.utils import (post_process, correct_boxes)
from hdtrack.utils.object import Object
import numpy as np
from hdtrack.utils.utils import image2Modelinput
from hdtrack.utils.visualization import plot_results

class BaseSdeEmbed():
    def __init__(self,detection,extractor,track_class_names):
        self.detection = detection
        self.extractor = extractor
        self.track_class_names = track_class_names
        self.model_image_size = self.detection.model_image_size
        self.extractor_image_size=self.extractor.extractor_image_size
        self.class_names = self.detection.class_names
        self.type_modelinput = self.detection.type_modelinput
        self.is_letterbox_image=self.detection.is_letterbox_image

    def detect_one_image(self, origin_image, draw=True):
        images_data,origin_image_shape,origin_image = image2Modelinput(origin_image,self.detection.model_image_size,self.detection.is_letterbox_image,self.detection.type_modelinput)
        predictions = self.get_predictions(images_data, origin_image_shape,origin_image)
        prediction = predictions[0]
        if not draw:
            return prediction
        elif prediction == None:  # 无框直接返回原图
            return origin_image
        else:
            return plot_results(origin_image,self.detection.class_names, prediction)

    def get_predictions(self,images_data,origin_image_shape,origin_image):
        results=self.get_detections(images_data,origin_image_shape)
        results=self.get_embeddings(results,origin_image)
        return results

    def get_embeddings(self, results, origin_image):
        raise NotImplementedError


    def get_detections(self, images_data, origin_images_shape):
        outputs=self.detection.model_inference(images_data)
        # tensor(B,levels*H*W*A,5+numclasses),xywh
        batch_detections = post_process(outputs, conf_thres=self.detection.confidence,nms_thres=self.detection.iou)
        # list[ndarray([num_anchors, 7])*B] 7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_label
        # 若无预测框，则为None
        results = []
        for i, batch_detection in enumerate(batch_detections):
            if batch_detection is None:
                results.append(None)
            else:
                ltrbs,scores,labels=batch_detection
                if self.detection.is_letterbox_image:
                    # 将box映射回origin_image
                    ltrbs = correct_boxes(ltrbs, self.detection.model_image_size,origin_images_shape[i])
                else:
                    ltrbs[:, [0, 2]] = np.clip(ltrbs[:, [0, 2]] / self.detection.model_image_size[0]*origin_images_shape[i][0],a_min=0,a_max=origin_images_shape[i][0])
                    ltrbs[:, [1, 3]] = np.clip(ltrbs[:, [1, 3]] / self.detection.model_image_size[1]*origin_images_shape[i][1],a_min=0,a_max=origin_images_shape[i][1])
                result = {'track':[],'untrack':[]}
                for ltrb,score,label in zip(ltrbs,scores,labels):
                    obj=Object(ltrb,'ltrb',label,score)
                    if self.detection.class_names[label] in self.track_class_names:
                        result['track'].append(obj)
                    else:
                        result['untrack'].append(obj)

                results.append(result)
        return results








