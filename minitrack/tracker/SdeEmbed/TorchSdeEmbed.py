import numpy as np
import torch
from .BaseSdeEmbed import BaseSdeEmbed
from minitrack.utils.utils import image2Modelinput
from minitrack.detection import  TorchDetection
from minitrack.extractor import TorchExtractor

class TorchSdeEmbed(BaseSdeEmbed):
    def __init__(self,track_class_names,detection=None,extrackor=None):
        self.detection=detection if detection is not None else TorchDetection()
        self.extractor=extrackor if extrackor is not None else TorchExtractor()
        super(TorchSdeEmbed, self).__init__(self.detection, self.extractor, track_class_names)

    def get_embeddings(self, results, origin_images):
        image_crops = []
        for result,origin_image in zip(results,origin_images):
            if result is None:
                continue
            for obj in result['track']:
                x1, y1, x2, y2 = obj.ltrb.astype(np.int32)
                crop = origin_image[y1:y2, x1:x2, :]
                crop, _, _ = image2Modelinput(crop, self.extractor.extractor_image_size, self.extractor.is_letterbox_image,
                                              self.extractor.type_modelinput)
                image_crops.append(crop)
        if len(image_crops)==0:
            return [[]]*len(origin_images)
        image_crops = torch.cat(image_crops,dim=0)
        embeddings = self.extractor(image_crops)

        i=0
        outputs=[]
        for result in results:
            if result is None :
                outputs.append([])
                continue
            for obj in result['track']:
                obj.feature=embeddings[i].cpu().numpy()
                i+=1
            outputs.append(result['track']+result['untrack'])
        return outputs

    def get_embed_tid(self, images, tids):
        # 输入的是由dataloader导入的，都是tensor格式
        if images.numel()==0:
            return None,None
        embeddings = self.extractor(images) #(B,embed_dim)
        tids=tids.to(self.extractor.device)
        return embeddings,tids #(B,1)











