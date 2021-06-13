import torch


def bbox_area(bbox):
    w = torch.clamp((bbox[..., 2] - bbox[..., 0]), min=0)
    h = torch.clamp((bbox[..., 3] - bbox[..., 1]), min=0)
    result = w * h
    return result


def bbox_intersection(bbox_1, bbox_2):
    inter_x1 = torch.maximum(bbox_1[..., 0], bbox_2[..., 0]).unsqueeze(-1)
    inter_y1 = torch.maximum(bbox_1[..., 1], bbox_2[..., 1]).unsqueeze(-1)
    inter_x2 = torch.minimum(bbox_1[..., 2], bbox_2[..., 2]).unsqueeze(-1)
    inter_y2 = torch.minimum(bbox_1[..., 3], bbox_2[..., 3]).unsqueeze(-1)
    return torch.cat((inter_x1, inter_y1, inter_x2, inter_y2), -1)


def bbox_iou(bbox_1, bbox_2):
    size_1 = bbox_1.size(0)
    size_2 = bbox_2.size(0)
    bbox_1 = bbox_1.expand(size_2, -1, -1).transpose(0, 1)
    bbox_2 = bbox_2.expand(size_1, -1, -1)
    areas_1 = bbox_area(bbox_1)
    areas_2 = bbox_area(bbox_2)
    bbox_inter = bbox_intersection(bbox_1, bbox_2)
    areas_inter = bbox_area(bbox_inter)
    result = areas_inter / (areas_1 + areas_2 - areas_inter)
    result = result
    return result


if __name__ == '__main__':
    bbox_1 = torch.tensor([[0, 0, 2, 2], [2, 1, 4, 3]], dtype=torch.float32)
    bbox_2 = torch.tensor([[1, 1, 3, 3], [5, 5, 6, 6]], dtype=torch.float32)
    area = bbox_area(bbox_1)
    iou = bbox_iou(bbox_1, bbox_2)
    print(area)
    print(iou)
    print(iou.size())
