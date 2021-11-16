import numpy as np
from scipy.spatial.distance import cdist
import lap
from minitrack.utils.np_util import np_box_iou
from .kalman_filter import chi2inv95


def min_cost_matching(distance_metric, thresh, tracks, detectobjs, track_indices=None,detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detectobjs)))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(tracks, detectobjs, track_indices, detection_indices)
    _, row_indices, col_indices = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    track_indices = np.asarray(track_indices)
    detection_indices = np.asarray(detection_indices)
    unmatched_tracks = track_indices[np.where(row_indices<0)[0]].tolist()
    unmatched_detections = detection_indices[np.where(col_indices < 0)[0]].tolist()
    
    matches=[]
    for irow, icol in enumerate(row_indices):
        if icol >= 0:
            track_idx = track_indices[irow]
            detection_idx = detection_indices[icol]
            if cost_matrix[irow, icol] > thresh:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))

    #print('unmatched_detections',unmatched_detections)
    return matches, unmatched_tracks, unmatched_detections

def matching_cascade(distance_metric, thresh, cascade_depth, tracks, detections,track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []

    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break
        track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0: # Nothing to match at this level
            continue
        matches_l, _, unmatched_detections = min_cost_matching(distance_metric, thresh, tracks, detections,track_indices_l, unmatched_detections)
        matches += matches_l

    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections

def iou_cost(tracks, detections, track_indices,detection_indices):

    det_ltrb = np.array([detections[i].ltrb for i in detection_indices])
    track_ltrb = np.array([tracks[i].ltrb for i in track_indices])

    ious=np_box_iou(track_ltrb,det_ltrb)
    cost_ious=1-ious
    return cost_ious

def embedding_cost(tracks, detections, track_indices,detection_indices):
    det_feat = np.array([detections[i].feature for i in detection_indices])
    track_feat = np.array([tracks[i].smooth_feat for i in track_indices])
    cost_matrix = np.maximum(0.0, cdist(track_feat, det_feat))
    # cdist函数这里默认是欧式距离
    # 因为这里embedding在前向传播时经过了F.normalize，所以欧式距离等于1-cosine
    return cost_matrix

def gate_cost_funct(kf,cost_matrix,tracks,detections,track_indices,detection_indices,only_position=False,lambda_=0.98):
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].xyah for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        # gating_distance是马氏距离
        # cost_matric是外貌特征
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position,metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix




