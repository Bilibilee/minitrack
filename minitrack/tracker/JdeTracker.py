from .BaseTracker import BaseTracker
from .utils.matching import *

class JdeTracker(BaseTracker):
    def __init__(self, embed_model,max_iou_distance_first=0.5,max_iou_distance_second=0.7,max_cosine_distance=0.7,max_age=30, n_init=1,budget=30):
        super(JdeTracker, self).__init__(embed_model,max_age,n_init,budget)

        self.max_iou_distance_first = max_iou_distance_first
        self.max_iou_distance_second = max_iou_distance_second
        self.max_cosine_distance = max_cosine_distance
        self.track_class_names = self.embed_model.track_class_names

    def _match(self, detectobjs):

        def gated_cost(tracks, detections, track_indices, detection_indices):
            cost_matrix = embedding_cost(tracks, detections, track_indices, detection_indices)
            cost_matrix = gate_cost_funct(self.kalman_filter, cost_matrix, tracks, detections, track_indices, detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = min_cost_matching(gated_cost, self.max_cosine_distance,self.tracks, detectobjs, track_indices=confirmed_tracks)
        #print('unmatched_detections',unmatched_detections)
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates =  [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections =min_cost_matching(iou_cost, self.max_iou_distance_first, self.tracks,detectobjs, iou_track_candidates, unmatched_detections)

        matches_c, unmatched_tracks_c, unmatched_detections = min_cost_matching(iou_cost, self.max_iou_distance_second,self.tracks, detectobjs,unconfirmed_tracks, unmatched_detections)

        matches = matches_a + matches_b+matches_c

        unmatched_tracks = list(unmatched_tracks_a + unmatched_tracks_b+unmatched_tracks_c)
        return matches, unmatched_tracks, unmatched_detections






