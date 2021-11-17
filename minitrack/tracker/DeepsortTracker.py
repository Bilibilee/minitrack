from .BaseTracker import BaseTracker
from .utils.matching import *

class DeepsortTracker(BaseTracker):
    def __init__(self, embed_model,max_iou_distance=0.7,max_cosine_distance=0.7,max_age=70, n_init=3,budget=30):
        super(DeepsortTracker, self).__init__(embed_model,max_age,n_init,budget)

        self.max_iou_distance = max_iou_distance
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
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(gated_cost, self.max_cosine_distance, self.max_age,self.tracks, detectobjs, track_indices=confirmed_tracks)
        #print('unmatched_detections',unmatched_detections)
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections =min_cost_matching(iou_cost, self.max_iou_distance, self.tracks,detectobjs, iou_track_candidates, unmatched_detections)
        matches = matches_a + matches_b
        unmatched_tracks = list(unmatched_tracks_a + unmatched_tracks_b)
        return matches, unmatched_tracks, unmatched_detections






