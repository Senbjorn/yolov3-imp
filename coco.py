import torch
from collections import defaultdict


def load_classes(names_path):
    with open(names_path, 'r') as names_file:
        names = [line.strip() for line in names_file]
    return names


def compute_name_id_mappings(images):
    name_id = {}
    id_name = {}
    for img_rec in images:
        img_name = img_rec['file_name']
        img_id = img_rec['id']
        name_id[img_name] = img_id
        id_name[img_id] = img_name
    return name_id, id_name


def compute_coco_detections(preds, image_id):
    result = []
    preds = preds.cpu().detach().numpy()
    for pred in preds:
        x = pred[0]
        y = pred[1]
        width = pred[2] - x
        height = pred[3] - y
        bbox = [x, y, width, height]
        bbox = list(map(lambda x: round(float(x), 2), bbox))
        category_id = int(pred[6]) + 1
        score = float(pred[4])
        coco_pred = {
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'score': score
        }
        result.append(coco_pred)
    return result


def map_ids(coco_dt, id_mapping):
    for dt in coco_dt:
        dt['category_id'] = id_mapping[dt['category_id']]
    return coco_dt


def convert_coco_detections(annotations, id_mapping):
    _, id_name = compute_name_id_mappings(annotations['images'])
    detections = defaultdict(list)
    for a in annotations['annotations']:
        image_id = a['image_id']
        bbox = a['bbox']
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        bbox = torch.tensor(bbox)
        category_id = a['category_id']
        image_name = id_name[image_id]
        if category_id not in id_mapping:
            continue
        else:
            category_id = id_mapping[category_id]
        data = torch.zeros(1, 5, dtype=torch.float32)
        data[0, :4] = bbox
        data[0, 4] = category_id
        detections[image_name].append(data)
    detections_cat = {}
    for image_name in detections:
        data = torch.cat(detections[image_name])
        detections_cat[image_name] = data
    return detections_cat
