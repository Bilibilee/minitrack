from hdtrack.detection.BaseDetection import BaseDetection
import onnxruntime as ort

class OnnxDetection(BaseDetection):
    def __init__(self,cfg_path='cfg/yolov4_cfg.json'):
        super(OnnxDetection, self).__init__(cfg_path,'ndarray')


    def initialize(self,cfg):
        print('Loading weights from : '+cfg['onnx_model_path'])
        self.model= ort.InferenceSession(cfg['onnx_model_path'], providers=['CUDAExecutionProvider']) # 创建一个推理session
        print('Finished!')

    def model_inference(self,images_data):
        # images_data is numpy array on cpu
        # X is numpy array on cpu
        '''
        input_ortvalue = ort.OrtValue.ortvalue_from_numpy(images_data, 'cuda', 0)
        io_binding = self.model.io_binding()
        self.model.run(None,{'input':input_ortvalue})
        io_binding.bind_input(name='input', device_type=input_ortvalue.device_name(), device_id=0, element_type=np.float32,
                              shape=input_ortvalue.shape(), buffer_ptr=input_ortvalue.data_ptr())
        io_binding.bind_output('output')
        self.model.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()[0]
        '''
        inputname=self.model.get_inputs()[0].name
        outputs=self.model.run(None,{inputname:images_data})
        return outputs
