import numpy as np
class Object():
    def __init__(self, box,box_type,label,score,embed=None,track_id=None):
        if box_type!='ltrb' and box_type!='ltwh':
            raise ValueError("box_type must be 'ltrb' or 'ltwh' ")

        if box_type=='ltrb':
            self.ltrb=np.asarray(box,dtype=np.float32)
            self.ltwh=np.asarray( self.ltrb_to_ltwh(box), dtype=np.float32)
        else:
            self.ltwh=np.asarray(box,dtype=np.float32)
            self.ltrb=np.asarray(self.ltwh_to_ltrb(box),dtype=np.float32)
        self.label = int(label)
        self.score = score
        self.feature=embed

        self.track_id= track_id

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self,feature):
        if feature is not None:
            feature = np.asarray(feature, dtype=np.float32)
            feature /= np.linalg.norm(feature)
            self._feature = feature# 默认L2范式,在做L2归一化,有利于embed效果!!!!
        else:
            self._feature = None

    @classmethod
    def generate_Objects(cls,prediction):
        '''
        results=[]
        for object in zip(prediction['ltrbs'],prediction['scores'],prediction['labels'],prediction['embeds']):
            ltrb,score,label,embed=object
            dt=cls(ltrb,label,score,embed)
            results.append(dt)
        return results
        '''
        pass

    @property
    def xyah(self):
        ret = self.ltwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def ltrb_to_ltwh(ltrb):
        ret = np.asarray(ltrb).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def ltwh_to_ltrb(ltwh):
        ret = ltwh.copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        if self.track_id:
            return str(self.ltrb)+',label:%d,score:%2f,track_id:%d'%(self.label,self.score,self.track_id)
        else:
            return str(self.ltrb)+',label:%d,score:%2f'%(self.label,self.score)
