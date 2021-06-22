import cv2
import numpy as np


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
