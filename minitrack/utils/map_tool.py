import torch
import numpy as np
import os
import matplotlib.pyplot as plt

class Map_Analysis():

    def __init__(self,results_files_path,IOU_threshold,class_names):
        self.class_names=class_names
        self.results_files_path=results_files_path
        self.IOU_threshold=IOU_threshold
        self.gts=[] # gts:为dict组成的list，dict记录每张图片的信息，dict的key为classlabel，value为tensor(n,4)
        self.gts_used = [] # gts_used为dict组成的list，dict记录每张图片的信息，dict的key为classlabel，value为n个0组成的list。0表示bbox还没有被匹配过
        self.predictions=[] #记录每张的predbbox信息，还是str类型，img_id,4*bboc,score,classlabel
        self.count_gt_class={} # count_gt_class:统计gt中各个类别有多少bbox
        self.classlabels=None  # classlabels:统计gt中出现的所有类别，label从0开始标注
        self.load_result_txt()


    def load_result_txt(self):
        labels=[]

        with open(self.results_files_path+'/result.txt') as f:
            for i,line in enumerate(f.readlines()):
                if i % 2==1:
                    self.gts.append(line.strip().split()) #img_id,4*bbox,classlabel
                else:
                    self.predictions.append(line.strip().split()) #img_id,4*bboc,conf,classlabel
            for line in self.gts:
                labels.append([bbox.split(',')[-1] for bbox in line[1:]])
        # count_gt_class:统计gt中各个类别有多少bbox
        # classlabels:统计gt中出现的所有类别，从0开始标注
        for label in labels:
            for item in label:
                self.count_gt_class[int(item)] = self.count_gt_class.get(int(item), 0) + 1
        self.classlabels = sorted(self.count_gt_class.keys())

        for i in range(len(self.gts)):
            bboxes = [bbox for bbox in self.gts[i][1:]]
            class_bboxes = {}
            gt_used = {}
            for bbox in bboxes:

                bbox = list(map(int, bbox.split(',')))
                label = bbox[-1]
                bbox = bbox[:4]
                if label not in class_bboxes:
                    class_bboxes[label] = torch.tensor(bbox, dtype=torch.float).reshape(-1, 4)
                    gt_used[label] = [0]
                else:
                    tensor = torch.tensor(bbox, dtype=torch.float).reshape(-1, 4)
                    class_bboxes[label] = torch.cat([class_bboxes[label], tensor], dim=0)
                    gt_used[label].extend([0])
            # gts:为dict组成的list，dict记录每张图片的信息，dict的key为classlabel，value为tensor(n,4)
            # gts_used为dict组成的list，dict记录每张图片的信息，dict的key为classlabel，value为n个0组成的list。0表示bbox还没有被匹配过
            self.gts[i] = class_bboxes
            self.gts_used.append(gt_used)

    def box_area(self,boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def compute_iou(self,pred_bbox,gt):
        # 输入是一个预测框和多个gt
        # 返回与该预测框iou最大的gt的下标索引和iou值
        lt=torch.max(pred_bbox[:,:2],gt[:,:2])
        rb=torch.min(pred_bbox[:,2:],gt[:,2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:,0] * wh[:, 1]
        iou=inter/(self.box_area(pred_bbox)+self.box_area(gt)-inter)

        return torch.max(iou,dim=0)

    def voc_ap(self,rec, prec):
        #通过召回率和精确度，计算每个类别的值ap，就是计算PR图下的面积
        rec.insert(0, 0.0) # insert 0.0 at begining of list
        rec.append(1.0) # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0) # insert 0.0 at begining of list
        prec.append(0.0) # insert 0.0 at end of list
        mpre = prec[:]

        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])

        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i)

        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre


    def compute_map(self,draw_plot):
        count_tp={} #只需要记录tp。fp=len(pred_bbox)-tp
        pred_counter_per_class={}
        sum_AP=0
        ap_dictionary={}
        for classlabel in self.classlabels:
            pred_bboxes=[] #存放所有类别为classlabel的预测框，x1,y1,x2,y2,score,label,img_id
            for prediction in self.predictions:
                img_id=int(prediction[0])

                for pred in prediction[1:]:
                    pred=pred.split(',')
                    bbox=list(map(int,pred[:4]))
                    score=float(pred[4])
                    try:
                        label=int(pred[5])
                    except:
                        print(img_id)
                    if label == classlabel:
                        bbox.extend([score,label,img_id])
                        pred_bboxes.append(bbox)


            pred_bboxes=sorted(pred_bboxes,key=lambda x:-x[4]) # 按score降序排序
            pred_counter_per_class[classlabel] = len(pred_bboxes)

            tp=[0]*len(pred_bboxes) #len(pred_bboxes)等于tp+fp
            fp=[0]*len(pred_bboxes)
            score=[0]*len(pred_bboxes)

            print('Start compute AP: '+self.class_names[classlabel])

            for i,pred_bbox in enumerate(pred_bboxes):
                img_id=pred_bbox[-1]
                bbox=torch.tensor(pred_bbox[:4],dtype=torch.float).reshape(-1,4)
                score[i]=pred_bbox[4]

                if classlabel not in self.gts[img_id - 1]: #该张图片没有这个类别的gt
                    fp[i]=1
                    continue
                else:
                    iou,index=self.compute_iou(bbox,self.gts[img_id - 1][classlabel])

                    if iou>=self.IOU_threshold and self.gts_used[img_id-1][classlabel][index]==False:
                        # 通过gts_used[img_id-1][classlabel][index]==False判断是否重复检测！！
                        tp[i]=1
                        count_tp[classlabel]=count_tp.get(classlabel,0)+1
                        self.gts_used[img_id-1][classlabel][index]=True
                    else:
                        fp[i]=1

            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]

            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / self.count_gt_class[classlabel]

            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

            ap, mrec, mprec = self.voc_ap(rec[:], prec[:])
            F1 = np.array(rec) * np.array(prec) / (np.array(prec) + np.array(rec)+1e-6) * 2

            sum_AP += ap

            ap_dictionary[classlabel]=ap

            """
            画每个类别的AP、F1、Recall、Precision
            """
            if draw_plot:
                plt.plot(rec, prec, '-o')
                # add a new penultimate point to the list (mrec[-2], 0.0)
                # since the last line segment (and respective area) do not affect the AP value
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                # set window title
                fig = plt.gcf()  # gcf - get current figure
                fig.canvas.set_window_title('AP ' + self.class_names[classlabel])
                # set plot title
                AP_text = "{0:.2f}%".format(ap * 100) + " = " + self.class_names[classlabel] + " AP IOUthre="+str(self.IOU_threshold)
                F1_text = self.class_names[classlabel] + " F1 IOUthre="+str(self.IOU_threshold)
                Recall_text =  self.class_names[classlabel] + " Recall IOUthre="+str(self.IOU_threshold)
                Precision_text =  self.class_names[classlabel] + " Precision IOUthre="+str(self.IOU_threshold)


                plt.title('class: ' + AP_text)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                # save the plot
                if not os.path.exists(self.results_files_path + "/AP_iou={:.2f}".format(self.IOU_threshold)):
                    os.makedirs(self.results_files_path + "/AP_iou={:.2f}".format(self.IOU_threshold))
                fig.savefig(self.results_files_path + "/AP_iou={:.2f}/{}.png".format(self.IOU_threshold,self.class_names[classlabel]))
                plt.cla()  # clear axes for next plot

                plt.plot(score, F1, "-", color='orangered')
                plt.title('class: ' + F1_text )
                plt.xlabel('score')
                plt.ylabel('F1')
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                if not os.path.exists(self.results_files_path + "/F1_iou={:.2f}".format(self.IOU_threshold)):
                    os.makedirs(self.results_files_path + "/F1_iou={:.2f}".format(self.IOU_threshold))
                fig.savefig(self.results_files_path + "/F1_iou={:.2f}/{}.png".format(self.IOU_threshold,self.class_names[classlabel]))
                plt.cla()  # clear axes for next plot

                plt.plot(score, rec, "-H", color='gold')
                plt.title('class: ' + Recall_text )
                plt.xlabel('Score')
                plt.ylabel('Recall')
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                if not os.path.exists(self.results_files_path + "/Recall_iou={:.2f}".format(self.IOU_threshold)):
                    os.makedirs(self.results_files_path + "/Recall_iou={:.2f}".format(self.IOU_threshold))
                fig.savefig(self.results_files_path +"/Recall_iou={:.2f}/{}.png".format(self.IOU_threshold,self.class_names[classlabel]))
                plt.cla()  # clear axes for next plot

                plt.plot(score, prec, "-s", color='palevioletred')
                plt.title('class: ' + Precision_text )
                plt.xlabel('Score')
                plt.ylabel('Precision')
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                if not os.path.exists(self.results_files_path + "/Precision_iou={:.2f}".format(self.IOU_threshold)):
                    os.makedirs(self.results_files_path + "/Precision_iou={:.2f}".format(self.IOU_threshold))
                fig.savefig(self.results_files_path +"/Precision_iou={:.2f}/{}.png".format(self.IOU_threshold,self.class_names[classlabel]))
                plt.cla()  # clear axes for next plot

        return ap_dictionary,sum_AP,pred_counter_per_class,count_tp

    def adjust_axes(self,r, t, fig, axes):
        # get text width for re-scaling
        bb = t.get_window_extent(renderer=r)
        text_width_inches = bb.width / fig.dpi
        # get axis width in inches
        current_fig_width = fig.get_figwidth()
        new_fig_width = current_fig_width + text_width_inches
        propotion = new_fig_width / current_fig_width
        # get axis limit
        x_lim = axes.get_xlim()
        axes.set_xlim([x_lim[0], x_lim[1]*propotion])

    def draw_plot_func(self,dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar=None):
        # sort the dictionary by decreasing value, into a list of tuples
        sorted_dic_by_value = sorted(dictionary.items(), key=lambda x:x[1])
        # unpacking the list of tuples into two lists
        sorted_keys, sorted_values = zip(*sorted_dic_by_value)

        if true_p_bar !=None:  # 画detecion-results时要用，一个水平直方图分为tp和fp
            """
             Special case to draw in:
                - green -> TP: True Positives (object detected and matches ground-truth)
                - red -> FP: False Positives (object detected but does not match ground-truth)
                - orange -> FN: False Negatives (object not detected but present in the ground-truth)
            """
            fp_sorted = []
            tp_sorted = []
            for key in sorted_keys:

                fp_sorted.append(dictionary[key] - true_p_bar[key])
                tp_sorted.append(true_p_bar[key])
            plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive') # 水平直方图
            plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
            # add legend
            plt.legend(loc='lower right')
            """
             Write number on side of bar
            """
            fig = plt.gcf() # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                fp_val = fp_sorted[i]
                tp_val = tp_sorted[i]
                fp_str_val = " " + str(fp_val)
                tp_str_val = fp_str_val + " " + str(tp_val)
                # trick to paint multicolor with offset:
                # first paint everything and then repaint the first number
                t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
                plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
                if i == (len(sorted_values)-1): # largest bar
                    self.adjust_axes(r, t, fig, axes)
        else:
            plt.barh(range(n_classes), sorted_values, color=plot_color)
            """
             Write number on side of bar
            """
            fig = plt.gcf() # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                str_val = " " + str(val) # add a space before
                if val < 1.0:
                    str_val = " {0:.2f}".format(val)
                t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
                # re-set axes to show number inside the figure
                if i == (len(sorted_values)-1): # largest bar
                    self.adjust_axes(r, t, fig, axes)
        # set window title
        fig.canvas.set_window_title(window_title)
        # write classes in y axis
        tick_font_size = 12
        plt.yticks(range(n_classes), [self.class_names[label] for label in sorted_keys], fontsize=tick_font_size)
        """
         Re-scale height accordingly
        """
        init_height = fig.get_figheight()
        # comput the matrix height in points and inches
        dpi = fig.dpi
        height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
        height_in = height_pt / dpi
        # compute the required figure height
        top_margin = 0.15 # in percentage of the figure height
        bottom_margin = 0.05 # in percentage of the figure height
        figure_height = height_in / (1 - top_margin - bottom_margin)
        # set new height
        if figure_height > init_height:
            fig.set_figheight(figure_height)

        # set plot title
        plt.title(plot_title, fontsize=14)
        # set axis titles
        # plt.xlabel('classes')
        plt.xlabel(x_label, fontsize='large')
        # adjust size of window
        fig.tight_layout()
        # save the plot
        fig.savefig(output_path)
        # show image
        if to_show:
            plt.show()
        # close the plot
        plt.close()

