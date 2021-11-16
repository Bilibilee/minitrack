from .BaseJdeEmbed import BaseJdeEmbed
import onnxruntime as ort
import numpy as np


class OnnxJdeEmbed(BaseJdeEmbed):
    def __init__(self, track_class_names,cfg_path='cfg/jde_cfg.json'):
        super(OnnxJdeEmbed, self).__init__(track_class_names,cfg_path,'ndarray')

    def initialize(self, cfg):
        self.num_features = len(cfg['strides'])
        self.num_anchors = len(cfg['anchors_shape'][0])
        self.features_shape = [[self.model_image_size[0] // stride, self.model_image_size[1] // stride] for stride in cfg['strides']]

        self.embed_mask = self.generate_embed_mask()
        print('Loading weights from : ' + cfg['onnx_model_path'])
        self.model = ort.InferenceSession(cfg['onnx_model_path'], providers=['CUDAExecutionProvider'])  # 创建一个推理session
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


    def generate_embed_mask(self):
        # 用来embed的mask，因为embed是levels*H*W，所以是多个anchor共享一个embedding
        # 我们在nms需要用到embed_mask，nms是对levels*H*W*A个anchor进行nms，所以需要进行匹配
        # 返回embed_mask：(levels*H*W*A,)。embed_mask[i]==该anchor对应的embedding的下标，建立起(levels*H*W*A,)到(levels*H*W,)的映射
        embeds_mask = []

        start_index = 0
        for i in range(self.num_features):
            embed_mask = np.arange(start_index,start_index + self.features_shape[i][1] * self.features_shape[i][0])\
                .reshape(self.features_shape[i][1], self.features_shape[i][0]).astype(np.int32)
            # (H,W)
            embed_mask=np.ascontiguousarray(embed_mask)
            start_index += self.features_shape[i][1] * self.features_shape[i][0]

            embed_mask = np.tile(embed_mask[:,:,None],reps=(1, 1, self.num_anchors))
            # (H,W,A)
            embeds_mask.append(embed_mask.flatten())  # (H*W*A,)
        embeds_mask = np.concatenate(embeds_mask, axis=0)  # (levels*H*W*A,)
        return embeds_mask






