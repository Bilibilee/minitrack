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

    def __init__(self, ltwh, score, label, feat , kf ,max_age,n_init):

        self._ltwh = np.asarray(ltwh, dtype=np.float32)
        self.label = label
        self.score = score

        self.kalman_filter=kf
        self.mean, self.covariance =self.kalman_filter.initiate(self.ltwh_to_xyah(self._ltwh))

        self.state=TrackState.Tentative
        self.smooth_feat = None
        self.update_features(feat)
        self.track_id = self.next_id()  # stastic method

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

    def predict(self):
        self.time_since_update += 1
        #mean_state = self.mean.copy()
        #if self.time_since_update > 1:
        #    mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        #print('mean',self.mean,'covariance',self.covariance)

    def update(self, detecttrack):
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, detecttrack.xyah)

        self.score = detecttrack.score
        self.update_features(detecttrack.feature)
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


