import os
import shutil
import torch
from collections import defaultdict

from yolov3.data import save_index
from yolov3.utils import json_load, print_verbose


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


def convert_coco_annotations(annotations, id_mapping):
    ann_tensor = defaultdict(list)
    for a in annotations['annotations']:
        annotation_id = a['id']
        image_id = a['image_id']
        bbox = a['bbox']
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        bbox = torch.tensor(bbox)
        category_id = a['category_id']
        if category_id not in id_mapping:
            continue
        else:
            category_id = id_mapping[category_id]
        data = torch.zeros(1, 6, dtype=torch.float32)
        data[0, :4] = bbox
        data[0, 4] = category_id
        data[0, 5] = annotation_id
        ann_tensor[image_id].append(data)
    ann_tensor_cat = {}
    for image_id in ann_tensor:
        data = torch.cat(ann_tensor[image_id])
        ann_tensor_cat[image_id] = data
    return ann_tensor_cat


def convert_coco_detections(detections, id_mapping):
    det_tensor = defaultdict(list)
    c = 1
    for d in detections:
        image_id = d['image_id']
        bbox = d['bbox']
        score = d['score']
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        bbox = torch.tensor(bbox)
        category_id = d['category_id']
        if category_id not in id_mapping:
            continue
        else:
            category_id = id_mapping[category_id]
        data = torch.zeros(1, 7, dtype=torch.float32)
        data[0, :4] = bbox
        data[0, 4] = score
        data[0, 5] = category_id
        data[0, 6] = c
        det_tensor[image_id].append(data)
        c += 1
    det_tensor_cat = {}
    for image_id in det_tensor:
        data = torch.cat(det_tensor[image_id])
        det_tensor_cat[image_id] = data
    return det_tensor_cat


def prepare_data(images_path, annotations_path, output_path, cat_id_mapping=None, verbose=True):
    print_func = print_verbose(verbose)
    print_func('create output dirs\n')
    output_images_path = os.path.join(output_path, 'images')
    output_annotations_path = os.path.join(output_path, 'annotations')
    os.mkdir(output_images_path)
    os.mkdir(output_annotations_path)
    annotations = json_load(annotations_path)
    images = annotations['images']
    _, index = compute_name_id_mappings(images)

    # copy images
    print_func('copy images\n')
    for c, image_id in enumerate(index):
        image_file = index[image_id]
        image_path = os.path.join(images_path, image_file)
        target_image_path = os.path.join(output_images_path, image_file)
        shutil.copyfile(image_path, target_image_path)
        print_func(f'done {c + 1:10d}/{len(index)}\r')
    print_func('\n')
    print_func('save images index\n')
    index_path = os.path.join(output_images_path, 'index.json')
    save_index(index, index_path)

    # convert annotations
    print_func('convert annotations\n')
    anns = annotations['annotations']
    anns_index = {}
    image_anns = defaultdict(list)
    print_func('indexing annotations\n')
    for a in anns:
        image_anns[a['image_id']].append(a)
    for c, image_id in enumerate(index):
        if image_id not in image_anns:
            bboxes = torch.empty(0, 6, dtype=torch.float32)
        else:
            bboxes = []
            for a in image_anns[image_id]:
                bbox = a['bbox']
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                bbox_id = a['id']
                category_id = a['category_id']
                if cat_id_mapping is not None:
                    category_id = cat_id_mapping[category_id]
                data = torch.tensor([*bbox, category_id, bbox_id], dtype=torch.float32).unsqueeze(0)
                bboxes.append(data)
            bboxes = torch.cat(bboxes)
        image_name = os.path.splitext(index[image_id])[0]
        bboxes_file = image_name + '.pt'
        bboxes_path = os.path.join(output_annotations_path, bboxes_file)
        torch.save(bboxes, bboxes_path)
        anns_index[image_id] = bboxes_file
        print_func(f'done {c + 1:10d}/{len(index)}\r')
    print_func('\n')
    print_func('save annotations index\n')
    anns_index_path = os.path.join(output_annotations_path, 'index.json')
    save_index(anns_index, anns_index_path)
