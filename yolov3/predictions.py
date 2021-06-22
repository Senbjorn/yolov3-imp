import torch
import numpy as np

from yolov3.bbox import bbox_iou


def compute_predictions(model_output, confidence, num_classes, nms_conf=0.4):
    box_corner = torch.zeros_like(model_output[:, :, :4])
    box_corner[:, :, 0] = (model_output[:, :, 0] - model_output[:, :, 2]/2)
    box_corner[:, :, 1] = (model_output[:, :, 1] - model_output[:, :, 3]/2)
    box_corner[:, :, 2] = (model_output[:, :, 0] + model_output[:, :, 2]/2)
    box_corner[:, :, 3] = (model_output[:, :, 1] + model_output[:, :, 3]/2)
    model_output[:, :, :4] = box_corner
    batch_size = model_output.size(0)
    output = []
    for ind in range(batch_size):
        res = []
        image_pred = model_output[ind]
        _, max_ind = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_ind = max_ind.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_ind)
        image_pred = torch.cat(seq, 1)
        if not (image_pred[:, 4] <= confidence).all().item():
            image_pred_ = image_pred[image_pred[:, 4] > confidence]
            image_classes = torch.unique(image_pred_[:, -1])
            for cls in image_classes:
                image_pred_class = image_pred_[image_pred_[:, -1] == cls]
                conf_sort_index = torch.sort(image_pred_class[:, 4], descending =True)[1]
                image_pred_class = image_pred_class[conf_sort_index]
                while image_pred_class.size(0) > 0:
                    res.append(image_pred_class[0].unsqueeze(0))
                    if image_pred_class.size(0) == 1:
                        break
                    ious = bbox_iou(image_pred_class[0, :4].unsqueeze(0), image_pred_class[1:, :4]).squeeze(0)
                    image_pred_class = image_pred_class[1:]
                    image_pred_class = image_pred_class[ious < nms_conf]
        if len(res) != 0:
            res = torch.cat(res, 0)
        else:
            res = None
        output.append(res)
    return output
