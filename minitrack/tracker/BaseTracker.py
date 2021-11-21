import cv2
import numpy as np
from .utils.kalman_filter import KalmanFilter
from .utils.track import Track
from minitrack.utils.visualization import plot_results
from minitrack.utils.object import Object


class BaseTracker:
    def __init__(self, embed_model,max_age, n_init,budget,max_save_image_num):
        self.max_age = max_age
        self.n_init = n_init
        self.budget=budget # centers轨迹跟踪的容量
        self.max_save_image_num=max_save_image_num
        self.embed_model=embed_model
        self.kalman_filter = KalmanFilter()
        self.tracks = []

    def clear(self):
        Track.count=0
        self.tracks = []

    def predict(self):
        for track in self.tracks:
            track.predict()

    def multi_predict(self):
        if len(self.tracks) > 0:
            multi_mean = np.asarray([t.mean for t in self.tracks])
            multi_covariance = np.asarray([t.covariance for t in self.tracks])
            multi_mean, multi_covariance = self.kalman_filter.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                self.tracks[i].mean = mean
                self.tracks[i].covariance = cov
                self.tracks[i].time_since_update+=1


    def initiate_track(self, obj):
        self.tracks.append(Track(obj.ltwh, obj.score, obj.label, obj.feature,obj.abnormal_class_image,self.kalman_filter,self.max_age,self.n_init,self.budget,self.max_save_image_num))

    def update(self, detectobjs):

        matches, unmatched_tracks, unmatched_detections = self._match(detectobjs)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detectobjs[detection_idx])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # 利用python内存管理机制去析构，Update distance metric.
        for detection_idx in unmatched_detections:
            self.initiate_track(detectobjs[detection_idx])


    def _match(self,detectobjs):
        raise NotImplementedError

    def track_one_image(self,origin_image,draw=True,detect_abnormal_behavior=True,save_path='./results',right_direction='y-',count_threshold=0.7,ignore_count_num=8):
        prediction=self.embed_model.detect_one_image(origin_image, draw=False)
        needtrack_objs=[obj for obj in prediction if obj.feature is not None ]
        unneedtrack_objs=[obj for obj in prediction if obj.feature is None]

        self.multi_predict()#卡尔曼滤波
        self.update(needtrack_objs)

        if detect_abnormal_behavior==True:
            import time
            localtime = time.localtime(time.time())
            str_localtime = time.strftime('%Y_%m_%d_%H_%M_%S', localtime)
            self.detect_NoHelmet(origin_image,str_localtime,save_path)
            self.detect_WrongDirection(origin_image,str_localtime,save_path,right_direction,count_threshold,ignore_count_num)

        outputs = []
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
                # box,box_type,label,score,embed=None,track_id=None):
            obj=Object(track.ltwh,'ltwh',track.label,track.score,track_id=track.track_id,centers=track.centers)

            outputs.append(obj)
        outputs.extend(unneedtrack_objs)
        if not draw:
            return outputs
        elif len(outputs) == 0:  # 无框直接返回原图
            return origin_image
        else:
            return plot_results(origin_image,self.embed_model.class_names, outputs)



    def detect_NoHelmet(self,origin_image,str_localtime,save_path):
        for track in self.tracks:
            if track.abnormal_class_image is  None or track.cur_NoHelmet_save_image_num >= track.max_save_image_num:
                continue
            track.cur_NoHelmet_save_image_num+=1
            save_image_name=save_path+'/'+str_localtime+'_NoHelmet_'+str(track.track_id)+'.jpg'
            # '%Y_%m_%d_%H_%M_%S_NoHelmet_trackid.jpg'
            abnormal_class_image= cv2.cvtColor(track.abnormal_class_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_image_name,abnormal_class_image)
            x1,y1,x2,y2=track.ltrb.astype(np.int32)

            save_image_name = save_path + '/' + str_localtime + '_NoHelmetMotor_' + str(track.track_id) + '.jpg'
            track_class_image = origin_image[y1:y2, x1:x2, :]
            track_class_image=cv2.cvtColor(track_class_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_image_name,track_class_image)

    def detect_RunRed(self):
        pass

    def detect_WrongDirection(self,origin_image,str_localtime,save_path,right_direction,count_threshold,ignore_count_num):
        if right_direction not in ['x+','x-','y+','y-']:
            raise ValueError("right_direcetion must in ['x+','x-','y+','y-']")
        for track in self.tracks:
            x_increase_num=0
            x_decrease_num=0
            y_increase_num=0
            y_decrease_num=0
            centers=track.centers
            num_centers=len(centers)

            if num_centers < ignore_count_num or track.cur_WrongDirect_save_image_num >= track.max_save_image_num:
                continue
            x1, y1, x2, y2 = track.ltrb.astype(np.int32)
            w=x2-x1
            h=y2-y1
            for i in range(1,num_centers):
                x_diff=centers[i][0] - centers[i-1][0]
                y_diff=centers[i][1] - centers[i-1][1]
                if x_diff > 0.03*w:
                    x_increase_num+=1
                elif x_diff<-0.03*w:
                    x_decrease_num+=1

                if y_diff>0.03*h:
                    y_increase_num+=1
                elif y_diff<-0.03*h:
                    y_decrease_num+=1

            if (right_direction=='x-' and x_increase_num/num_centers > count_threshold ) or \
               (right_direction=='x+' and x_decrease_num/num_centers > count_threshold) or \
               (right_direction=='y-' and y_increase_num/num_centers > count_threshold )   or \
               (right_direction=='y+' and y_decrease_num/num_centers > count_threshold):

                track.cur_WrongDirect_save_image_num += 1
                save_image_name = save_path + '/' + str_localtime + '_WrongDirect_' + str(track.track_id)+'.jpg'
                # '%Y_%m_%d_%H_%M_%S_WrongDirect_trackid.jpg'

                track_class_image = origin_image[y1:y2, x1:x2, :]
                track_class_image = cv2.cvtColor(track_class_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_image_name , track_class_image)













