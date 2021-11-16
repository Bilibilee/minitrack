from minitrack.nets.JDE import JDE
from .BaseJdeEmbed import BaseJdeEmbed
import torch
from minitrack.nets.loss.jde_loss import JDELoss
import json
from minitrack.utils.loss_util import train_funct
from minitrack.utils.dataset import MutiltaskDataset
import numpy as np

class TorchJdeEmbed(BaseJdeEmbed):
    def __init__(self,track_class_names,cfg_path='cfg/jde_cfg.json'):
        super(TorchJdeEmbed ,self).__init__(track_class_names,cfg_path,'tensor')
        self.output_names = ['xywh', 'det_conf', 'cls_conf', 'embeddings']

    def initialize(self,cfg):
        self.device = torch.device('cuda') if cfg['cuda'] else torch.device('cpu')
        anchors_shape = torch.tensor(cfg['anchors_shape'], dtype=torch.float32, requires_grad=False, device=self.device)
        self.num_features =len(cfg['strides'])
        self.num_anchors=len(anchors_shape[0])
        self.features_shape = [[self.model_image_size[0] // stride, self.model_image_size[1] // stride] for stride in cfg['strides']]

        self.model = JDE(cfg['embedding_dim'],cfg['strides'],self.class_names, anchors_shape,self.model_image_size, cfg['cuda'])
        self.embed_mask=self.generate_embed_mask()
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
        self.model.net.load_state_dict(model_dict)
        print('Finished!')

    def model_inference(self,images_data):
        self.model.net.eval()
        with torch.no_grad():
            images_data = images_data.to(self.device)
            outputs = self.model(images_data)
        return outputs

    def generate_embed_mask(self):
        # 用来embed的mask，因为embed是levels*H*W，所以是多个anchor共享一个embedding
        # 我们在nms需要用到embed_mask，nms是对levels*H*W*A个anchor进行nms，所以需要进行匹配
        # 返回embed_mask：(levels*H*W*A,)。embed_mask[i]==该anchor对应的embedding的下标，建立起(levels*H*W*A,)到(levels*H*W,)的映射
        embeds_mask = []

        start_index = 0
        for i in range(self.num_features):
            embed_mask = torch.arange(start_index, start_index + self.features_shape[i][1] * self.features_shape[i][0],
                                      requires_grad=False, device=self.device, dtype=torch.long) \
                                     .reshape(self.features_shape[i][1], self.features_shape[i][0])
            # (H,W)
            start_index += self.features_shape[i][1] * self.features_shape[i][0]

            embed_mask = embed_mask.unsqueeze(dim=2).repeat((1, 1, self.num_anchors))
            # (H,W,A)
            embeds_mask.append(embed_mask.flatten())  # (H*W*A,)
        embeds_mask = torch.cat(embeds_mask, dim=0)  # (levels*H*W*A,)
        return embeds_mask


    def torch2onnx(self,batchsize=1,save_onnx_path='cfg/jde.onnx'):
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
        train_detection=cfg['train_detection']
        train_embedding=cfg['train_embedding']

        train_detect_anno_path = cfg['train_detect_anno_path']
        train_embed_anno_path = cfg['train_embed_anno_path']
        val_detect_anno_path = cfg['test_detect_anno_path']
        val_embed_anno_path = cfg['test_embed_anno_path']

        with open(train_detect_anno_path) as f:
            train_detect_lines = f.readlines()
        with open(train_embed_anno_path) as f:
            train_embed_lines = f.readlines()
        with open(val_detect_anno_path) as f:
            val_detect_lines = f.readlines()
        with open(val_embed_anno_path) as f:
            val_embed_lines = f.readlines()


        train_dataset = MutiltaskDataset(train_detect_lines, train_embed_lines, self.model_image_size, mosaic=mosaic,is_random=is_random)  # 马赛克增强
        val_dataset=MutiltaskDataset(val_detect_lines, val_embed_lines, self.model_image_size, mosaic=False,is_random=False)
        # 建立loss函数
        jde_loss = JDELoss(train_detection, train_embedding,train_dataset.nID,self.model.embedding_dim,self.model.anchors, self.model.anchors_shape, self.model.num_classes,
                           self.model.num_anchors,self.model.num_features, self.model_image_size,self.device,label_smooth=smooth_label,mean=False)

        train_funct(self.model.net,self.start_epoch,self.device,self.cfg_path, jde_loss,  train_dataset, val_dataset)


    def get_embed_tid(self, images_data, targets):
        self.model.net.eval()
        with torch.no_grad():
            images_data = images_data.to(self.device)
            outputs = self.model.net(images_data)

        jde_loss = JDELoss(True, False, -1, self.model.embedding_dim,
                           self.model.anchors, self.model.anchors_shape, self.model.num_classes,
                           self.model.num_anchors, self.model.num_features, self.model_image_size, self.device)
        embedddings,tids=jde_loss.get_embed_tid(outputs,targets)
        return embedddings,tids



