import numpy as np
from .utils.kalman_filter import KalmanFilter
from .utils.track import Track
from hdtrack.utils.visualization import plot_results
from hdtrack.utils.object import Object

class BaseTracker:
    def __init__(self, embed_model,max_age, n_init):
        self.max_age = max_age
        self.n_init = n_init
        self.embed_model=embed_model
        self.kalman_filter = KalmanFilter()
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
        self.tracks.append(Track(obj.ltwh, obj.score, obj.label, obj.feature,self.kalman_filter,self.max_age,self.n_init))

    def update(self, detectobjs):

        matches, unmatched_tracks, unmatched_detections = self._match(detectobjs)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detectobjs[detection_idx])

        for track_idx in unmatched_tracks:
            is_delete=self.tracks[track_idx].mark_missed()
            if is_delete==True:
                self.tracks[track_idx]=None # 利用python智能指针去做析构
        self.tracks=list(filter(lambda x:x is not None,self.tracks))

        for detection_idx in unmatched_detections:
            self.initiate_track(detectobjs[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # Update distance metric.

    def _match(self,detectobjs):
        raise NotImplementedError

    def track_one_image(self,origin_image,draw=True):
        prediction=self.embed_model.detect_one_image(origin_image, draw=False)
        needtrack_objs=[obj for obj in prediction if obj.feature is not None ]
        unneedtrack_objs=[obj for obj in prediction if obj.feature is None]

        self.multi_predict()#卡尔曼滤波
        self.update(needtrack_objs)

        outputs = []
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
                # box,box_type,label,score,embed=None,track_id=None):
            obj=Object(track.ltwh,'ltwh',track.label,track.score,track_id=track.track_id)
            outputs.append(obj)
        outputs.extend(unneedtrack_objs)
        if not draw:
            return outputs
        elif len(outputs) == 0:  # 无框直接返回原图
            return origin_image
        else:
            return plot_results(origin_image,self.embed_model.class_names, outputs)
