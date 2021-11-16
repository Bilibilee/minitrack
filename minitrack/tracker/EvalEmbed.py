import numpy as np
from minitrack.utils.dataset import YoloDataset,yolo_dataset_collate,EmbedDataset,embed_dataset_collate
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
from .JdeEmbed import TorchJdeEmbed
from .SdeEmbed import TorchSdeEmbed


class EvalEmbed():
    def __init__(self,embed):
        self.embed=embed

    def get_embed_tid(self,anno_path,Batchsize):
        with open(anno_path) as f:
            lines = [line.strip() for line in f.readlines()]

        if isinstance(self.embed,TorchJdeEmbed):
            testdataset = YoloDataset(lines, self.embed.model_image_size, mosaic=False, is_random=False)
            collate_fn=yolo_dataset_collate
        elif isinstance(self.embed,TorchSdeEmbed):
            testdataset=EmbedDataset(lines,self.embed.extractor_image_size)
            collate_fn=embed_dataset_collate
        else:
            raise TypeError('type must be TorchJdeEmbed or TorchSdeEmbed')

        gen = DataLoader(testdataset, batch_size=Batchsize, collate_fn=collate_fn
            , pin_memory=True,drop_last=True,shuffle=False)

        embeds = []
        tids = []
        with tqdm(total=len(testdataset) // Batchsize) as pbar:
            for batch in gen:
                images, targets = batch

                embed, tid = self.embed.get_embed_tid(images, targets)
                if embed != None and tid != None:
                    embeds.append(embed)
                    tids.append(tid)
                pbar.update(1)
        embeds = torch.cat(embeds, dim=0)
        tids = torch.cat(tids, dim=0)
        return embeds,tids

    def test_roc(self,anno_path,Batchsize,save_path = './results'):

        embeds,tids=self.get_embed_tid(anno_path,Batchsize)
        n = len(tids)
        assert n == len(embeds)
        embeds = F.normalize(embeds, dim=1)
        pdist = torch.mm(embeds, embeds.t()).cpu().numpy()
        # pdist，就是每个向量之间的余弦相似度，越趋于1时，说明预测这两个向量越相似

        tids = tids.expand(n, n).eq(tids.expand(n, n).t()).cpu().numpy()
        # gt取值0、1，1就是这两个id一样。

        up_triangle = np.where(np.triu(pdist) - np.eye(n) * pdist != 0)
        # up_triangle上三角，不包括对角线
        # 问，万一有个pdist=0呢，概率很低，不会正好等于0，浮点数嘛
        pdist = pdist[up_triangle]
        tids = tids[up_triangle]

        # far, tar, threshold = metrics.roc_curve(tids, pdist)
        fpr, tpr, thresholds = metrics.roc_curve(tids, pdist)
        roc = metrics.auc(fpr, tpr)

        fig = plt.gcf()
        plt.plot(fpr, tpr)
        plt.title('ROC=' + str(roc), fontsize=14)
        plt.show()
        fig.savefig(save_path + "/ROC.png")

        # 去算假阳性，真阳性
        print('roc:{}'.format(roc))
        return roc

