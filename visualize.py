import cv2
import torch
import numpy as np


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


def uniform_color_palette(num_classes):
    hues = np.linspace(0, 160, num=num_classes).astype(np.uint8)
    palette = np.full((num_classes, 3), 255, dtype=np.uint8)
    palette[:, 0] = hues
    palette = palette[np.newaxis]
    palette = cv2.cvtColor(palette, cv2.COLOR_HSV2RGB)
    return palette


def draw_palette(palette, height=40, width=10):
    img = palette.repeat(width, axis=1)
    img = np.tile(img, (height, 1, 1))
    return img


def draw_bbox(img, bbox, label, color, linewidth=1, fontscale=1., fontthickness=1):
    c1 = tuple(bbox[0:2].int().numpy())
    c2 = tuple(bbox[2:4].int().numpy())
    img = cv2.rectangle(img, c1, c2, color, linewidth)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, fontscale, fontthickness)[0]
    if c1[1] - t_size[1] >= 0:
        c1 = c1[0], c1[1] - t_size[1]
    c2 = c1[0] + t_size[0], c1[1] + t_size[1]
    img = cv2.rectangle(img, c1, c2, color, -1)
    img = cv2.putText(img, label, (c1[0], c1[1] + t_size[1]), cv2.FONT_HERSHEY_PLAIN, fontscale, [0,0,0], fontthickness)
    return img


def draw_predictions(img, bboxes, palette, classes):
    img_bbox = img.copy()
    for bbox in bboxes:
        cls = int(bbox[-1])
        color = list(map(int, palette[0, cls])) # wtf?? doesn't work with np types
        label = classes[cls]
        scale = max(*img.shape[:2]) / 416
        width = max(1, int(scale))
        thickness = max(1, int(scale))
        img_bbox = draw_bbox(
            img_bbox, bbox.cpu(), label, color,
            linewidth=width, fontscale=scale, fontthickness=thickness
        )
    return img_bbox


def img_to_tensor(img):
    img = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0) / 255
    return img
