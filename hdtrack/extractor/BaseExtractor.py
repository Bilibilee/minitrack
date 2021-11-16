import json

class BaseExtractor():
    def __init__(self,cfg_path):
        cfg = json.load(open(cfg_path))
        self.cfg_path = cfg_path
        self.extractor_image_size = cfg['extractor_image_size']
        self.is_letterbox_image = cfg['is_letterbox_image']
        self.initialize(cfg)


    def initialize(self, cfg):
        raise  NotImplementedError

    def __call__(self, images_data):
        raise  NotImplementedError













