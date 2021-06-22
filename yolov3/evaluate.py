import os

import torch
from torch.utils.data import DataLoader

from yolov3.predictions import compute_predictions
from yolov3.data import EvalDataset, aspect_transform, clamp_results, inv_aspect_transform_results, read_index, save_index
from yolov3.utils import print_verbose


def evaluate(yolo, input_path, output_path, num_classes, confidence=0.25, nms_threshold=0.4, batch_size=1, device='cpu', verbose=True):
    print_func = print_verbose(verbose)
    
    print_func('read index\n')
    width = yolo.net_info['width']
    height = yolo.net_info['height']
    dataset = EvalDataset(input_path, width, height)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    output_index = {}
    yolo.to(device)
    yolo.eval()
    pred_id_counter = 1

    counter = 0
    total = len(dataset)
    
    print_func('evaluation\n')
    with torch.no_grad():
        for X, info in dataloader:
            X = X.to(device)
            preds = yolo(X)
            results = compute_predictions(preds, confidence, num_classes, nms_conf=nms_threshold)
            for i in range(X.size(0)):
                result = results[i]
                image_id = int(info[i, 0].item())
                img_w, img_h = info[i, 1].item(), info[i, 2].item()
                if result is None:
                    result = torch.empty(0, 7, dtype=torch.float32)
                else:
                    num_bboxes = result.size(0)
                    asp = aspect_transform(img_w, img_h, width, height)
                    result = inv_aspect_transform_results(result, asp)
                    result = clamp_results(result, img_w, img_h)
                    result_ids = torch.arange(pred_id_counter, pred_id_counter + num_bboxes).unsqueeze(1)
                    pred_id_counter += num_bboxes
                    result = torch.cat((result.cpu(), result_ids), dim=1)
                file_name = os.path.splitext(dataset.index[image_id])[0]
                output_file = file_name + '.pt'
                output_file_path = os.path.join(output_path, output_file)
                output_index[image_id] = output_file
                torch.save(result, output_file_path)
            counter += X.size(0)
            print_func(f'done {counter:10d}/{total}\r')
    print_func('\n')
    print_func('save index\n')
    output_index_path = os.path.join(output_path, 'index.json')
    save_index(output_index, output_index_path)
