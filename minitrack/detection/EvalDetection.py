from minitrack.utils.utils import (post_process,image2Modelinput)
from minitrack.utils.visualization import plot_results
import numpy as np
from minitrack.utils.dataset import YoloDataset,yolo_dataset_collate
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
import cv2
from minitrack.utils.map_tool import Map_Analysis
import warnings

class EvalDetection():
    def __init__(self,detection):
        self.detection=detection

        self.model_image_size = self.detection.model_image_size
        self.class_names = self.detection.class_names
        self.confidence = self.detection.confidence
        self.iou = self.detection.iou
        self.is_letterbox_image = self.detection.is_letterbox_image

    def test_fps(self, origin_image, test_interval):
        # 输入为一张图片，PIL格式
        images_data, origin_images_shape,_ = image2Modelinput(origin_image, self.model_image_size, self.is_letterbox_image,type_modelinput=self.detection.type_modelinput)
        model_inference_time = 0
        nms_time = 0
        print('-----------------------------------')
        print('  begin test FPS')
        with tqdm(total=test_interval) as pbar:
            for i in range(test_interval):
                t1=time.time()
                outputs=self.detection.model_inference(images_data)
                # tensor(B,levels*H*W*A,5+numclasses),xywh
                t2 = time.time()
                post_process(outputs, conf_thres=self.confidence, nms_thres=self.iou)
                t3 = time.time()
                model_inference_time += t2 - t1
                nms_time += t3 - t2
                pbar.update()

        model_inference_time /= test_interval
        nms_time /= test_interval

        print(' model inference time: %f s' % model_inference_time)
        print('       nms time      : %f s' % nms_time)
        print('       FPS           : %f' % (1.0/(nms_time+model_inference_time)))
        print('-----------------------------------')

        return model_inference_time, nms_time


    def predict_images(self,anno_path,Batchsize,save_path,save_image_interval):
        with open(anno_path) as f:
            lines = [line.strip() for line in f.readlines()]
        if self.detection.confidence > 0.01:
            warnings.warn(' warning: testing map make sure confidence low enough ')

        testdataset = YoloDataset(lines, self.model_image_size, mosaic=False, is_random=False)
        gen = DataLoader(testdataset, batch_size=Batchsize,collate_fn=yolo_dataset_collate,pin_memory=True, drop_last=True)

        if not os.path.exists(save_path + '/images/'):
            os.makedirs(save_path + '/images/')

        f = open(save_path + '/result.txt', 'w')

        img_id = 1
        with tqdm(total=len(testdataset) // Batchsize) as pbar:
            for batch in gen:
                images_data, targets = batch
                images_data=images_data if self.detection.type_modelinput=='tensor' else images_data.cpu().numpy()
                predictions = self.detection.get_predictions(images_data, [self.model_image_size] * Batchsize)
                # 这里只是测验map，为了简单，就不映射回原图像了，所以origin_image_shape=model_image_size*Batchsize
                # 返回list[ dict{'ltrb':tensor(n,4),'labels':tensor(n),'scores':tensor(n)} * B ]
                # 注意labels为整数，且labels和scores都是一维的tensor。boxes为x1,y1,x2,y2
                # 当没有预测框时，为[]

                for prediction, target, image_data in zip(predictions, targets, images_data):
                    target = target if isinstance(target,np.ndarray) else target.cpu().numpy()
                    image = image_data if isinstance(image_data,np.ndarray) else image_data.cpu().numpy()
                    image = (image * 255.0).astype(np.uint8) # must be uint8
                    image = np.transpose(image, (1,2,0))

                    # 记录predict结果到result.txt
                    if len(prediction)==0:
                        # 如果没有检测出物体
                        f.write(str(img_id))
                    else:
                        f.write(str(img_id))
                        for obj in prediction:
                            listprint = obj.ltrb.astype(np.int32)
                            listprint = map(str, listprint)
                            listprint = ','.join(listprint)
                            f.write(' ' + listprint + ',' + str(obj.score) + ',' + str(obj.label))
                    f.write('\n')

                    # 记录groud-truth到result.txt
                    f.write(str(img_id))
                    if len(target) == 0:
                        img_id += 1
                        f.write('\n')
                        continue
                    gt = target[:, :5]
                    gt[:, 0], gt[:, 2] = gt[:, 0] - gt[:, 2] / 2, gt[:, 0] + gt[:, 2] / 2
                    gt[:, 1], gt[:, 3] = gt[:, 1] - gt[:, 3] / 2, gt[:, 1] + gt[:, 3] / 2
                    # 把xywh转换为左上角右下角
                    for i in range(len(gt)):
                        listprint = gt[i].astype(np.int32)
                        listprint = map(str, listprint)
                        listprint = ','.join(listprint)
                        f.write(' ' + listprint)
                    f.write('\n')

                    # 保存图片
                    if img_id % save_image_interval == 0 and len(prediction)!=0:  # 每隔20张存储
                        image = plot_results(image, self.class_names, prediction)
                        cv2.imwrite(save_path + "/images/" + str(img_id) + ".jpg", image)
                    img_id += 1

                pbar.update(1)

    def map_analysis(self,class_names,save_path,IOU_threshold,draw_plot):
        map = Map_Analysis(save_path,IOU_threshold,class_names)

        ap_dictionary, sum_AP, pred_counter_per_class, count_tp = map.compute_map(draw_plot)

        if len(count_tp) < len(map.classlabels):
            for label in map.classlabels:
                if label not in count_tp:
                    count_tp[label] = 0

        if len(pred_counter_per_class) < len(map.classlabels):
            for label in map.classlabels:
                if label not in count_tp:
                    pred_counter_per_class[label] = 0

        """
        统计各个类别有多少gt框
        """
        if draw_plot:
            window_title = "ground-truth-info"
            plot_title = "ground-truth\n"
            plot_title += "(" + str(len(map.gts)) + " files and " + str(len(map.classlabels)) + " classes)"
            x_label = "Number of objects per class"
            output_path = map.results_files_path + "/ground-truth-info.png"
            to_show = False
            plot_color = 'forestgreen'
            map.draw_plot_func(
                map.count_gt_class,
                len(map.classlabels),
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
            )

        """
        统计每个类别的预测框有多少个
        """

        if draw_plot:
            window_title = "detection-results-info"
            # Plot title
            plot_title = "detection-results\n"
            plot_title += "(" + str(len(map.gts)) + " files and "
            count_non_zero_values_in_dictionary = sum(
                int(x) > 0 for x in list(pred_counter_per_class.values()))  # 可能有的类别根本就没有检测到
            plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
            # end Plot title
            x_label = "Number of objects per class"
            output_path = map.results_files_path + "/detection-results-info.png"
            to_show = False
            plot_color = 'forestgreen'

            true_p_bar = count_tp

            map.draw_plot_func(
                pred_counter_per_class,
                len(pred_counter_per_class),
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                true_p_bar
            )

        """
         Draw mAP plot (Show AP's of all classes in decreasing order)
        """
        if draw_plot:
            window_title = "mAP"
            mAP = sum_AP / len(map.classlabels)
            plot_title = "mAP = {0:.2f}".format(mAP * 100)
            x_label = "Average Precision"
            output_path = map.results_files_path + "/mAP_iouthreshold={:.2f}.png".format(map.IOU_threshold)

            to_show = True
            plot_color = 'royalblue'
            map.draw_plot_func(
                ap_dictionary,
                len(map.classlabels),
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color
            )

    def test_map(self,anno_path,Batchsize,save_path='./results',save_image_interval=20,draw_plot=True,IOU_threshold=0.5):
        self.predict_images(anno_path, Batchsize,save_path, save_image_interval)
        self.map_analysis(self.class_names,save_path, IOU_threshold, draw_plot)
