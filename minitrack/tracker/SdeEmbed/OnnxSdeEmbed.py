import numpy as np
from .BaseSdeEmbed import BaseSdeEmbed
from minitrack.utils.utils import image2Modelinput
from minitrack.extractor import OnnxExtractor
from minitrack.detection import OnnxDetection


class OnnxSdeEmbed(BaseSdeEmbed):
    def __init__(self, track_class_names,detection=None,extrackor=None):
        self.detection = detection if detection is not None else OnnxDetection()
        self.extractor = extrackor if extrackor is not None else OnnxExtractor()
        super(OnnxSdeEmbed, self).__init__(self.detection,self.extractor,track_class_names)

    def get_embeddings(self, results, origin_image):
        image_crops = []

        for result in results:
            if result == None:
                continue
            for obj in result['track']:
                x1, y1, x2, y2 = obj.ltrb.astype(np.int32)
                crop = origin_image[y1:y2, x1:x2, :]
                crop, _, _ = image2Modelinput(crop, self.extractor.extractor_image_size,
                                              self.extractor.is_letterbox_image,
                                              self.extractor.type_modelinput)
                image_crops.append(crop)

        image_crops = np.concatenate(image_crops, axis=0)
        embeddings = self.extractor(image_crops)

        i = 0
        outputs = []
        for result in results:
            if result == None:
                continue
            for obj in result['track']:
                obj.feature = embeddings[i]
                i += 1
            outputs.append(result['track'] + result['untrack'])
        return outputs
















