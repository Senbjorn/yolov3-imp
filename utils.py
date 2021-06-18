import torch
import numpy as np
from bbox import bbox_iou


def predict_transform(prediction, input_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:,:,2:4]) * anchors
    prediction[:,:,:4] *= stride
    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    box_corner = torch.zeros_like(prediction[:, :, :4])
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_corner
    batch_size = prediction.size(0)
    output = []
    for ind in range(batch_size):
        res = []
        image_pred = prediction[ind]
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        if (image_pred[:, 4] <= confidence).all().item():
            continue
        image_pred_ = image_pred[image_pred[:, 4] > confidence]
        image_classes = torch.unique(image_pred_[:, -1])
        for cls in image_classes:
            image_pred_class = image_pred_[image_pred_[:,-1] == cls]
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending =True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            bboxes = image_pred_class.size(0)
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
