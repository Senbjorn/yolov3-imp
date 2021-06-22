import torch
import numpy as np

from yolov3.bbox import bbox_area, bbox_iou


def ignore_indices(ind, ignore):
    '''
    ind - array[int], [m]
    ignore - array[0|1], [m]
    '''
    new_inds = torch.zeros_like(ind)
    head = ignore == 0
    tail = ignore == 1
    n = head.sum()
    new_inds[:n] = ind[head]
    new_inds[n:] = ind[tail]
    return new_inds


def area_range_mask(bboxes, area_range):
    areas = bbox_area(bboxes)
    selection = (areas >= area_range[0]) & (areas <= area_range[1])
    return selection


def area_range_indices(bboxes, area_range):
    '''
    bboxes - array[[x1, y1, x2, y2]], [n_targets * 5]
    area_range - (min, max)
    '''
    ind = torch.arange(bboxes.size(0))
    ignore = torch.zeros_like(ind)
    if area_range is not None:
        selection = area_range_mask(bboxes, area_range)
        ignore[~selection] = 1
    ind = ignore_indices(ind, ignore)
    return ind, ignore[ind]


def bbox_match(preds, targets, ious, iou_thresholds=[0.5], num_preds=None, area_range=None):
    '''
    preds - array[[x1, y1, x2, y2, conf]], [n_preds * 5]
    targets - array[[x1, y1, x2, y2]], [n_targets * 4]
    ious - array[array[float]], [n_preds * n_targets]
    area_range - (min, max)
    '''
    indices = np.argsort(-preds[:, 4].detach().cpu().numpy(), kind='mergesort')
    indices = torch.from_numpy(indices)
    indices = indices[:num_preds]
    confidences = preds[indices, 4]
    # confidences, indices = torch.sort(preds[:, 4], descending=True)
    target_indices, target_ignore = area_range_indices(targets, area_range)

    num_thresholds = len(iou_thresholds)
    num_targets = len(target_indices)
    num_preds = len(indices)
    targets_matched = torch.full((num_thresholds, num_targets), -1)
    preds_matched = torch.full((num_thresholds, num_preds), -1)
    preds_mask = torch.zeros((num_thresholds, num_preds))
    for k, t in enumerate(iou_thresholds):
        for i, p_id in enumerate(indices.tolist()):
            j_m = -1
            iou = min(t, 1 - 1e-10)
            for j, t_id in enumerate(target_indices.tolist()):
                if targets_matched[k, j] > -1:
                    continue
                if j_m > -1 and target_ignore[j_m] == 0 and target_ignore[j] == 1:
                    break
                if ious[p_id, t_id] < iou:
                    continue
                j_m = j
                iou = ious[p_id, t_id]
            if j_m == -1:
                continue
            targets_matched[k, j_m] = p_id
            preds_matched[k, i] = target_indices[j_m]
            preds_mask[k, i] = target_ignore[j_m]

    bboxes = preds[indices, :4]
    mask = ~area_range_mask(bboxes, area_range)
    mask = mask.expand(num_thresholds, num_preds)
    mask = mask & (preds_matched == -1)
    preds_mask[mask] = 1
    
    result = {
        'num_preds': num_preds,
        'area_range': area_range,
        'pred_inds': indices,
        'target_inds': target_indices,
        'target_ignore': target_ignore,
        'confidences': confidences,
        'preds_matched': preds_matched,
        'targets_matched': targets_matched,
        'preds_mask': preds_mask,
    }
    return result


def compute_interpolated_precision(precision, recall, recall_thresholds):
    '''
    precision - array[], [n]
    recall - array[], [n]
    recall_thresholds - array[], [n_thr]
    '''
    interpolated_precision = torch.zeros_like(recall_thresholds)
    n = precision.size(0)
    max_precision = precision.flip(0).cummax(0).values.flip(0)
    inds = torch.searchsorted(recall, recall_thresholds, right=False)
    inds = inds[inds < n]
    interpolated_precision[:inds.size(0)] = max_precision[inds]
    return interpolated_precision


def loss_yolo(preds, targets):
    pass
