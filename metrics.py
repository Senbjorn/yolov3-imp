
import torch
from bbox import bbox_area, bbox_iou
import numpy as np


def get_area_range_indices(bboxes, area_range):
    '''
    bboxes - array[[x1, y1, x2, y2]], [n_targets * 5]
    area_range - (min, max)
    '''
    ind = torch.arange(bboxes.size(0))
    if area_range is not None:
        areas = bbox_area(bboxes)
        selection = (areas >= area_range[0])
        selection &= (areas <= area_range[1])
        ind = torch.where(selection)[0]
    return ind


def bbox_match(preds, targets, ious, iou_threshold=0.5, n_preds=None, area_range=None):
    '''
    preds - array[[x1, y1, x2, y2, conf]], [n_preds * 5]
    targets - array[[x1, y1, x2, y2]], [n_targets * 4]
    ious - array[array[float]], [n_preds * n_targets]
    area_range - (min, max)
    '''
    indices = np.argsort(-preds[:, 4].detach().cpu().numpy(), kind='mergesort')
    indices = torch.from_numpy(indices)
    indices = indices[:n_preds]
    confidences = preds[indices, 4]
    # confidences, indices = torch.sort(preds[:, 4], descending=True)
    target_indices = get_area_range_indices(targets, area_range)

    targets_matched = {}
    preds_matched = {}
    for p_id in indices:
        p_id = p_id.item()
        t_m = -1
        iou = min(iou_threshold, 1 - 1e-10)
        for t_id in target_indices:
            t_id = t_id.item()
            if t_id in targets_matched:
                continue
            if ious[p_id, t_id] < iou:
                continue
            t_m = t_id
            iou = ious[p_id, t_id]
        if t_m == -1:
            continue
        targets_matched[t_m] = p_id
        preds_matched[p_id] = t_m
    
    result = {
        'index': indices,
        'confidence': confidences,
        'match': torch.tensor([preds_matched[i.item()] if i.item() in preds_matched else -1 for i in indices], dtype=torch.int64),
    }

    bboxes = preds[indices]
    bboxes = bboxes[:, :4]
    unmached_indices = get_area_range_indices(bboxes, area_range)
    unmached_filter = (result['match'] != -1)
    unmached_filter[unmached_indices] = True
    result = {k: v[unmached_filter] for k, v in result.items()}
    
    result['num_targets'] = target_indices.size(0)
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


def compute_metrics(index, match, confidence, num_targets, recall_thresholds=None):
    '''
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    num_targets = TP + FN
    '''
    result = {
        'precision': None,
        'recall': None,
        'AP': None,
        'TP': None,
        'FP': None,
        'num_targets': num_targets,
    }
    if num_targets == 0:
        return result
    tp = (match >= 0).cumsum(0)
    fp = (match < 0).cumsum(0)
    precision = tp / (tp + fp)
    recall = tp / num_targets
    result['recall'] = recall
    result['precision'] = precision
    result['TP'] = tp
    result['FP'] = fp
    if recall_thresholds is None:
        recall_thresholds = torch.linspace(0., 1., 101)
    interpolated_precision = compute_interpolated_precision(precision, recall, recall_thresholds)
    average_precision = interpolated_precision.mean(0)
    result['AP'] = average_precision
    return result


def bbox_metrics(preds, targets, iou_threshold=0.5, n_preds=None, area_range=None, recall_thresholds=None):
    '''
    preds - array[[x1, y1, x2, y2, conf]], [n_preds * 5]
    targets - array[[x1, y1, x2, y2]], [n_targets * 4]
    '''
    ious = bbox_iou(preds[:, :4], targets[:, :4])
    match = bbox_match(
        preds[:, :5], targets[:, :4], ious,
        iou_threshold=iou_threshold,
        n_preds=n_preds,
        area_range=area_range
    )
    metrics = compute_metrics(
        match['index'],
        match['match'],
        match['confidence'],
        targets.size(0),
        recall_thresholds=recall_thresholds
    )
    return metrics


def loss_yolo(preds, targets):
    pass