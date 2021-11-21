import numpy as np

class TrackState(object):
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track():
    count=0
    @staticmethod
    def next_id():
        Track.count += 1
        return Track.count

    def __init__(self, ltwh, score, label, feat , abnormal_class_image, kf , max_age , n_init , budget,max_save_image_num):

        self._ltwh = np.asarray(ltwh, dtype=np.float32)
        self.label = label
        self.score = score
        self.track_id = self.next_id()  # stastic method
        self.kalman_filter=kf
        self.state = TrackState.Tentative

        self.mean, self.covariance =self.kalman_filter.initiate(self.ltwh_to_xyah(self._ltwh))

        self.smooth_feat = None
        self.update_features(feat)

        self.centers = [(int(self.mean[0]),int(self.mean[1]))]
        self.budget = budget

        self.cur_NoHelmet_save_image_num = 0
        self.cur_WrongDirect_save_image_num = 0
        self.cur_RunRed_save_image_num = 0
        self.max_save_image_num = max_save_image_num

        self.abnormal_class_image=None
        self.update_abnormal_class_image(abnormal_class_image)

        self.alpha = 0.9
        self.hits = 1
        self.time_since_update = 0
        self._n_init = n_init
        self._max_age = max_age

    def update_features(self, feat):
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            # 指数加权平均
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)  # 默认2范式

    def update_centers(self,center):
        self.centers.append(center)
        if self.budget is not None:
            self.centers = self.centers[-self.budget:]

    def update_abnormal_class_image(self,abnormal_class_image):
        if abnormal_class_image is None:
            return False
        if self.cur_NoHelmet_save_image_num >= self.max_save_image_num:
            self.abnormal_class_image = None
            return False
        self.abnormal_class_image = abnormal_class_image
        return True

    def predict(self):
        self.time_since_update += 1

        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)


    def update(self, detectobj):
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        self.score = detectobj.score
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, detectobj.xyah)

        self.update_centers((int(self.mean[0]), int(self.mean[1])))
        self.update_features(detectobj.feature)
        self.update_abnormal_class_image(detectobj.abnormal_class_image)

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

        if self.state==TrackState.Deleted:
            return True
        else:
            return False

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted


    @property
    def ltwh(self):
        # mean是xyah格式的
        # 左上角和wh
        if self.mean is None:
            return self._ltwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def ltrb(self):
        # 左上角和右下角
        ret = self.ltwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xyah(self):
        if self.mean is not None:
            return self.mean.copy()
        return self.ltwh_to_xyah(self.ltwh)

    @staticmethod
    def ltwh_to_xyah(ltwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(ltwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def ltbr_to_ltwh(ltrb):
        ret = np.asarray(ltrb).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def ltwh_to_ltrb(ltwh):
        ret = np.asarray(ltwh).copy()
        ret[2:] += ret[:2]
        return ret


