from minitrack.utils.utils import (post_process_embed, correct_boxes,match_abnormal_and_track)
from minitrack.utils.object import Object
import numpy as np
from minitrack.detection import BaseDetection

class BaseJdeEmbed(BaseDetection):
    def __init__(self,track_class_name,abnormal_class_name,cfg_path,type_modelinput):
        super(BaseJdeEmbed ,self).__init__(track_class_name,abnormal_class_name,cfg_path,type_modelinput)

        self.embed_mask=self.generate_embed_mask()


    def initialize(self, cfg):
        raise NotImplementedError

    def model_inference(self, images_data):
        raise NotImplementedError

    def generate_embed_mask(self):
        raise  NotImplementedError

    def get_predictions(self, images_data, origin_images_shape,origin_images=None):
        outputs=self.model_inference(images_data)
        # tensor(B,levels*H*W*A,5+numclasses),xywh
        batch_detections = post_process_embed(outputs,self.embed_mask, conf_thres=self.confidence,nms_thres=self.iou)
        # list[ndarray([num_anchors, 7])*B] 7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_label
        # 若无预测框，则为None
        results = []
        for i, batch_detection in enumerate(batch_detections):
            if batch_detection is None:
                results.append([])
            else:
                ltrbs,scores,labels,embeds=batch_detection
                if self.is_letterbox_image:
                    # 将box映射回origin_image
                    ltrbs = correct_boxes(ltrbs, self.model_image_size,origin_images_shape[i])
                else:
                    ltrbs[:, [0, 2]] = np.clip(ltrbs[:, [0, 2]] / self.model_image_size[0]*origin_images_shape[i][0],a_min=0,a_max=origin_images_shape[i][0])
                    ltrbs[:, [1, 3]] = np.clip(ltrbs[:, [1, 3]] / self.model_image_size[1]*origin_images_shape[i][1],a_min=0,a_max=origin_images_shape[i][1])

                result = {'track': [], 'abnormal': [], 'other': []}
                for ltrb, score, label,embed in zip(ltrbs, scores, labels,embeds):
                    obj = Object(ltrb, 'ltrb', label, score)
                    if self.class_names[obj.label] == self.track_class_name:
                        obj.embed=embed
                        result['track'].append(obj)
                    elif self.class_names[obj.label] == self.abnormal_class_name:
                        result['abnormal'].append(obj)
                    else:
                        result['other'].append(obj)

                if origin_images is not None:
                    result = match_abnormal_and_track(result, origin_images[i],self.match_iou_threshold)

                result = result['track'] + result['abnormal'] + result['other']
                results.append(result)

        return results









