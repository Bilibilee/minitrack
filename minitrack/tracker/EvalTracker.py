from minitrack.utils.utils import image2Modelinput
import numpy as np
from minitrack.utils.dataset import TrackerDataset,tracker_dataset_collate
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
import cv2
from minitrack.utils.visualization import plot_results
import motmetrics as mm
from .JdeEmbed import BaseJdeEmbed
from .SdeEmbed import BaseSdeEmbed

class EvalTracker():
    def __init__(self,tracker):
        self.tracker=tracker

    def test_fps(self, origin_image, test_interval):
        # 输入为一张图片，PIL格式
        images_data, origin_images_shape, origin_image = image2Modelinput(origin_image,
          self.tracker.embed_model.model_image_size,
          self.tracker.embed_model.is_letterbox_image,
          type_modelinput=self.tracker.embed_model.type_modelinput)
        detect_time = 0
        embed_time = 0
        match_time = 0
        print('-----------------------------------')
        print('  begin test FPS')
        with tqdm(total=test_interval) as pbar:
            for i in range(test_interval):
                t1 = time.time()
                if isinstance(self.tracker.embed_model,BaseJdeEmbed):
                    predictions=self.tracker.embed_model.get_predictions(images_data, origin_images_shape)
                    t2=time.time()
                elif isinstance(self.tracker.embed_model,BaseSdeEmbed):
                    results = self.tracker.embed_model.get_detections(images_data, origin_images_shape)
                    t2 = time.time()
                    predictions = self.tracker.embed_model.get_embeddings(results, origin_image)
                t3 = time.time()
                prediction = predictions[0]
                track_objs = [obj for obj in prediction if obj.feature is not None]
                untrack_objs = [obj for obj in prediction if obj.feature is None]
                self.tracker.predict()  # 卡尔曼滤波
                self.tracker.update(track_objs)
                t4 = time.time()

                detect_time += t2 - t1
                embed_time += t3 - t2
                match_time += t4 - t3
                pbar.update()

        detect_time /= test_interval
        embed_time /= test_interval
        match_time /= test_interval
        if isinstance(self.tracker.embed_model,BaseJdeEmbed):
            print('det_embed time : %f s' % (detect_time + embed_time))
        else:
            print(' detect time : %f s' % detect_time)
            print('  embed time : %f s' % embed_time)
        print('  match time : %f s' % match_time)
        print('      FPS    : %f ' % (1.0 / (detect_time + embed_time + match_time)))
        print('-----------------------------------')

    def write_results(self,filename, results, data_type='mot'):
        if data_type == 'mot':
            save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        else:
            raise ValueError(data_type)
        with open(filename, 'w') as f:
            for frame_id, ltwhs, track_ids in results:

                for ltwh, track_id in zip(ltwhs, track_ids):
                    if track_id < 0:
                        continue

                    x1, y1, w, h = ltwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    f.write(line)

    def test_mota(self,anno_path,save_path='./results/',iou_threshold=0.5):
        # 图片的地址全称的倒数第二个为分组
        # 此外要求组内图片有序！
        # batchsize 必须为1
        Batchsize=1
        with open(anno_path) as f:
            lines = [line.strip() for line in f.readlines()]

        split_lines = {}
        for line in lines:
            key = (line.split()[0]).split('\\')[-2]
            if key in split_lines:
                split_lines[key].append(line)
            else:
                split_lines[key] = [line]

        for key, each_lines in split_lines.items():
            print('eval '+key)
            if not os.path.exists(save_path + key):
                os.makedirs(save_path + key)

            testdataset = TrackerDataset(each_lines)
            gen = DataLoader(testdataset, batch_size=Batchsize, pin_memory=True, drop_last=True,collate_fn=tracker_dataset_collate)
            self.tracker.clear()
            frame_id = 1
            hypotheses = []
            gt = []
            with tqdm(total=len(testdataset) // Batchsize) as pbar:
                for batch in gen:

                    for origin_image, target in zip(*batch):
                        if target.size==0:
                            continue
                        objs=self.tracker.track_one_image(origin_image,draw=False)
                        online_ltwhs=[]
                        online_ids=[]
                        for obj in objs:
                            if obj.track_id is None:
                                continue
                            online_ltwhs.append(obj.ltwh)
                            online_ids.append(obj.track_id)

                        hypotheses.append((frame_id, online_ltwhs, online_ids))

                        ltwhs = target[:, :4]  # shape为(-1,6) ltwh,cls,id
                        gt.append((frame_id, ltwhs, target[:,-1].astype(np.int)))

                        result=plot_results(origin_image, self.tracker.embed_model.class_names, objs)
                        # cv2.imshow("video",frame)
                        cv2.imwrite(save_path + key + '/' + str(frame_id) + '.jpg', result)
                        frame_id += 1
                        pbar.update(1)

            self.write_results(save_path + key + '/hyp.txt', hypotheses)
            self.write_results(save_path + key + '/gt.txt', gt)

        accs = []
        for key in split_lines.keys():
            gt = mm.io.loadtxt(save_path + key + '/gt.txt', fmt="mot15-2D", min_confidence=1)  # 读入GT
            hyp = mm.io.loadtxt(save_path + key + '/hyp.txt', fmt="mot15-2D")  # 读入自己生成的跟踪结果

            acc = mm.utils.compare_to_groundtruth(gt, hyp, 'iou',distth=iou_threshold)
            # 根据GT和自己的结果，生成accumulator，distth是距离阈值
            accs.append(acc)

        mh = mm.metrics.create()
        summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=list(split_lines.keys()))
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)

