import torch
from hdtrack.nets.yolov4 import YOLOv4
from hdtrack.nets.loss.yolo_loss import YOLOLoss
import json
from hdtrack.utils.dataset import YoloDataset
import numpy as np
from hdtrack.detection.BaseDetection import BaseDetection
from hdtrack.utils.loss_util import train_funct

class TorchDetection(BaseDetection):
    def __init__(self,cfg_path='cfg/yolov4_cfg.json'):
        super(TorchDetection, self).__init__(cfg_path,'tensor')
        self.output_names=['xywh','det_conf','cls_conf']

    def initialize(self,cfg):
        self.device = torch.device('cuda' if cfg['cuda'] else 'cpu')
        anchors_shape = torch.tensor(cfg['anchors_shape'], dtype=torch.float32, requires_grad=False, device=self.device)
        self.model = YOLOv4(self.class_names, anchors_shape, self.model_image_size, cfg['cuda'])

        print('Loading weights from : '+cfg['torch_model_path'])
        model_dict = self.model.net.state_dict()
        if cfg['resume'] == True:
            checkpoint = torch.load(cfg['torch_model_path'], map_location=self.device)
            pretrained_dict = checkpoint['model']
            self.start_epoch = checkpoint['epoch']
        else:
            pretrained_dict = torch.load(cfg['torch_model_path'], map_location=self.device)
            self.start_epoch = 0
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
       # torch.save(model_dict,'cfg/yolov4_weights_helmet_final.pth')
        self.model.net.load_state_dict(model_dict)
        print('Finished!')

    def model_inference(self,images_data):
        self.model.net.eval()
        with torch.no_grad():
            images_data = images_data.to(self.device)
            outputs = self.model(images_data)
        return outputs

    def torch2onnx(self,batchsize=1,save_onnx_path='cfg/yolov4.onnx'):
        w,h=self.model_image_size
        x = torch.randn(size=(batchsize, 3, h, w),dtype=torch.float32).to(self.device)
        self.model.eval() #!!!!!
        print('-'*20+'\n'+'begin export')
        torch.onnx.export(self.model, x, save_onnx_path,opset_version=11,input_names=['input'], output_names=self.output_names)
        print('finish export'+'\n'+'-' * 20)
        with torch.no_grad():
            torch_out = self.model(x)
        print('-'*20+'\n'+'begin eval')
        import onnxruntime
        ort_session = onnxruntime.InferenceSession(save_onnx_path)
        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name:x.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(torch_out[0].cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        print('-' * 20)


    def train(self):
        cfg = json.load(open(self.cfg_path))
        mosaic = cfg['mosaic']
        is_random = cfg['is_random']  # dataset的数据增强
        smooth_label = cfg['smooth_label']  # 0.001#设置平衡标签，默认为0

        yolo_loss = YOLOLoss(self.model.anchors, self.model.anchors_shape, self.model.num_classes,self.model.num_anchors,
                self.model.num_features, self.model.model_image_size, self.device,label_smooth=smooth_label, mean=False)
        train_annotation_path = cfg['train_detect_anno_path']
        val_annotation_path = cfg['test_detect_anno_path']

        with open(train_annotation_path) as f:
            train_lines = f.readlines()
        with open(val_annotation_path) as f:
            val_lines = f.readlines()


        train_dataset = YoloDataset(train_lines, self.model_image_size, mosaic=mosaic, is_random=is_random)  # 马赛克增强
        val_dataset = YoloDataset(val_lines, self.model_image_size, mosaic=False, is_random=False)
        train_funct(self.model.net,self.start_epoch,self.device,self.cfg_path, yolo_loss, train_dataset, val_dataset)
