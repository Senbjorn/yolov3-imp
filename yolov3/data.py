import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from yolov3.utils import json_dump, json_load


def img_to_tensor(img):
    img = torch.from_numpy(img).float().permute(2,0,1) / 255
    return img


def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def aspect_transform(w, h, w_new, h_new):
    alpha = min(h_new / h, w_new / w)
    x_offset = (w_new - alpha * w) / 2
    y_offset = (h_new - alpha * h) / 2
    return alpha, x_offset, y_offset


def inv_aspect_transform_results(bboxes, asp):
    alpha, xo, yo = asp
    bboxes = bboxes.detach().clone()
    bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - xo) / alpha
    bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - yo) / alpha
    return bboxes


def clamp_results(bboxes, w, h):
    bboxes = bboxes.detach().clone()
    bboxes[:, [0, 2]] = torch.clamp(bboxes[:, [0, 2]], min=0, max=w)
    bboxes[:, [1, 3]] = torch.clamp(bboxes[:, [1, 3]], min=0, max=h)
    return bboxes


def resize_aspect(img, w, h):
    img_w = img.shape[1]
    img_h = img.shape[0]
    alpha, xo, yo = aspect_transform(img_w, img_h, w, h)
    new_w = int(img_w * alpha)
    new_h = int(img_h * alpha)
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((h, w, 3), 128)
    c1 = (int(xo), int(yo))
    c2 = (c1[0] + new_w, c1[1] + new_h)
    canvas[c1[1]: c2[1], c1[0]: c2[0], :] = new_img
    return canvas.astype(np.uint8)


def read_index(index_path):
    index_list = json_load(index_path)
    index = {r['image_id']: r['file_name'] for r in index_list}
    return index


def save_index(index, index_path):
    index_list = [{'image_id': k, 'file_name': v} for k, v in index.items()]
    json_dump(index_list, index_path)


class EvalDataset(Dataset):
    def __init__(self, data_path, width, height):
        self.data_path = data_path
        index_path = os.path.join(data_path, 'index.json')
        self.index = read_index(index_path)
        self.ids = sorted(list(self.index.keys()))
        self.width = width
        self.height = height

    def __getitem__(self, i):
        image_id = self.ids[i]
        file_name = self.index[image_id]
        img_path = os.path.join(self.data_path, file_name)
        img = read_img(img_path)
        img_w, img_h = img.shape[1], img.shape[0]
        img_info = torch.tensor([image_id, img_w, img_h])
        img_aspect = resize_aspect(img, self.width, self.height)
        inp = img_to_tensor(img_aspect)
        return inp, img_info

    def __len__(self):
        return len(self.ids)
