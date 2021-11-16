import numpy as np
from hdtrack.nets.extractor_model import Extractor
import torch
from .BaseExtractor import  BaseExtractor

class TorchExtractor(BaseExtractor):
    def __init__(self,cfg_path='cfg/extractor_cfg.json'):
        super(TorchExtractor,self).__init__(cfg_path)
        self.type_modelinput='tensor'

    def initialize(self, cfg):
        self.device = torch.device('cuda' if cfg['cuda'] else 'cpu')
        self.extractor=Extractor().to(self.device)

        print('Loading weights from : ' + cfg['torch_model_path'])
        model_dict = self.extractor.state_dict()
        pretrained_dict = torch.load(cfg['torch_model_path'], map_location=self.device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        self.extractor.load_state_dict(model_dict)
        print('Finished!')

    def __call__(self, images_data):
        self.extractor.eval()
        with torch.no_grad():
            images_data = images_data.to(self.device)
            outputs = self.extractor(images_data)
        return outputs

    def torch2onnx(self, batchsize=1,save_extractor_onnx_path='cfg/extractor.onnx'):
        w, h = self.extractor_image_size
        x = torch.randn(size=(batchsize, 3, h, w), dtype=torch.float32).to(self.device)
        self.extractor.eval()  # !!!!!
        print('-' * 20 + '\n' + 'begin export')
        torch.onnx.export(self.extractor, x, save_extractor_onnx_path, opset_version=11, input_names=['input'],
                          output_names=['embeddings'],dynamic_axes={'input':{0:'batch'},'embeddings':{0:'batch'}})
        print('finish export' + '\n' + '-' * 20)
        with torch.no_grad():
            torch_out = self.extractor(x)
        print('-' * 20 + '\n' + 'begin eval')
        import onnxruntime
        ort_session = onnxruntime.InferenceSession(save_extractor_onnx_path)
        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: x.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(torch_out.cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        print('-' * 20)













