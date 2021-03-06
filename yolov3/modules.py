import torch
from torch import nn
import numpy as np

from yolov3.config import ConfigParser


def transform_output(model_output, input_dim, anchors, num_classes):
    device = model_output.device
    batch_size = model_output.size(0)
    stride = input_dim // model_output.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    model_output = model_output.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    model_output = model_output.transpose(1, 2).contiguous()
    model_output = model_output.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    model_output[:, :, 0] = torch.sigmoid(model_output[:, :, 0])
    model_output[:, :, 1] = torch.sigmoid(model_output[:, :, 1])
    model_output[:, :, 4] = torch.sigmoid(model_output[:, :, 4])
    model_output[:, :, 5: 5 + num_classes] = torch.sigmoid((model_output[:,:, 5 : 5 + num_classes]))

    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)
    x_offset = torch.tensor(a, device=device, dtype=torch.float32).view(-1,1)
    y_offset = torch.tensor(b, device=device, dtype=torch.float32).view(-1,1)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    model_output[:, :, :2] += x_y_offset

    anchors = torch.tensor(anchors, device=device, dtype=torch.float32)
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    model_output[:, :, 2:4] = torch.exp(model_output[:,:,2:4]) * anchors
    model_output[:, :, :4] *= stride
    return model_output


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()
        if block['type'] == 'convolutional':
            activation = block['activation']
            filters= block["filters"]
            padding = block["pad"]
            kernel_size = block["size"]
            stride = block["stride"]
            if 'batch_normalize' in block:
                batch_normalize = block['batch_normalize']
            else:
                batch_normalize = 0
            bias = batch_normalize == 0
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            conv = nn.Conv2d(
                prev_filters, filters, kernel_size,
                stride, pad, bias=bias
            )
            module.add_module(f'conv_{index}', conv)
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f'batch_norm_{index}', bn)
            if activation == 'leaky':
                act = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f'leaky_{index}', act)
        if block['type'] == 'upsample':
            stride = block['stride']
            upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            module.add_module(f'upsample_{index}', upsample)
        if block['type'] == 'route':
            values = block['layers']
            start = values[0]
            if len(values) == 2:
                end = values[1]
            else:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module(f'route_{index}', route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
        if block['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{index}', shortcut)
        if block['type'] == 'yolo':
            mask = block['mask']
            anchors = block['anchors']
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)
            module.add_module(f'Detection_{index}', detection)
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info, module_list)


class Darknet(nn.Module):
    def __init__(self, cfg_path):
        super(Darknet, self).__init__()
        parser = ConfigParser()
        self.blocks = parser.parse(cfg_path)
        self.net_info, self.module_list = create_modules(self.blocks)
    
    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}
        detections = None
        for i, module in enumerate(modules):
            module_type = modules[i]['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
            elif module_type == 'route':
                layers = module['layers']
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    m1 = outputs[i + layers[0]]
                    m2 = outputs[i + layers[1]]
                    x = torch.cat((m1, m2), 1)
            elif module_type == 'shortcut':
                from_ = module["from"]
                x = outputs[i-1] + outputs[i+from_]
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                input_dim = self.net_info["height"]
                num_classes = module["classes"]
                x = x.detach()
                x = transform_output(x, input_dim, anchors, num_classes)
                if detections is None:
                    detections = x
                else:
                    detections = torch.cat((detections, x), 1)
            outputs[i] = x
        return detections


    def load_weights(self, weights_path):
        with open(weights_path, 'rb') as weights_file:
            header = np.fromfile(weights_file, dtype = np.int32, count = 5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(weights_file, dtype = np.float32)
        ptr = 0
        with torch.no_grad():
            for i in range(len(self.module_list)):
                module_type = self.blocks[i + 1]['type']
                if module_type == 'convolutional':
                    model = self.module_list[i]
                    conv = model[0]
                    if 'batch_normalize' in self.blocks[i + 1]:
                        batch_normalize = self.blocks[i + 1]['batch_normalize']
                    else:
                        batch_normalize = 0
                    if batch_normalize == 1:
                        bn = model[1]

                        #Get the number of weights of Batch Norm Layer
                        num_bn_biases = bn.bias.numel()

                        #Load the weights
                        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases

                        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases

                        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases

                        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases

                        #Cast the loaded weights into dims of model weights. 
                        bn_biases = bn_biases.view_as(bn.bias)
                        bn_weights = bn_weights.view_as(bn.weight)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                        bn_running_var = bn_running_var.view_as(bn.running_var)

                        #Copy the data to model
                        bn.bias.copy_(bn_biases)
                        bn.weight.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)
                    else:
                        #Number of biases
                        num_biases = conv.bias.numel()

                        #Load the weights
                        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                        ptr = ptr + num_biases

                        #reshape the loaded weights according to the dims of the model weights
                        conv_biases = conv_biases.view_as(conv.bias)

                        #Finally copy the data
                        conv.bias.copy_(conv_biases)
                    
                    num_weights = conv.weight.numel()

                    #Do the same as above for weights
                    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                    ptr = ptr + num_weights

                    conv_weights = conv_weights.view_as(conv.weight)
                    conv.weight.copy_(conv_weights)


class DarknetB(nn.Module):
    def __init__(self, cfg_path):
        super(DarknetB, self).__init__()
        parser = ConfigParser()
        self.blocks = parser.parse(cfg_path)
        self.net_info, self.module_list = create_modules(self.blocks)
    
    def forward(self, x):
        modules = self.blocks[1:]
        yolo_outputs = []
        outputs = {}
        for i, module in enumerate(modules):
            module_type = modules[i]['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
            elif module_type == 'route':
                layers = module['layers']
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    m1 = outputs[i + layers[0]]
                    m2 = outputs[i + layers[1]]
                    x = torch.cat((m1, m2), 1)
            elif module_type == 'shortcut':
                from_ = module["from"]
                x = outputs[i-1] + outputs[i+from_]
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                input_dim = self.net_info["height"]
                num_classes = module["classes"]
                yolo_outputs.append(
                    {
                        'anchors': anchors,
                        'num_classes': num_classes,
                        'input_dim': input_dim,
                        'output': x
                    }
                )
            outputs[i] = x
        return yolo_outputs


    def load_weights(self, weights_path):
        with open(weights_path, 'rb') as weights_file:
            header = np.fromfile(weights_file, dtype = np.int32, count = 5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(weights_file, dtype = np.float32)
        ptr = 0
        with torch.no_grad():
            for i in range(len(self.module_list)):
                block = self.blocks[i + 1]
                module_type = block['type']
                load_skip = 'load_skip' in block and block['load_skip']
                if module_type == 'convolutional':
                    model = self.module_list[i]
                    conv = model[0]
                    if 'batch_normalize' in block:
                        batch_normalize = block['batch_normalize']
                    else:
                        batch_normalize = 0
                    if batch_normalize == 1:
                        bn = model[1]

                        #Get the number of weights of Batch Norm Layer
                        num_bn_biases = bn.bias.numel()

                        #Load the weights
                        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases

                        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases

                        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases

                        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases

                        if not load_skip:
                            #Cast the loaded weights into dims of model weights. 
                            bn_biases = bn_biases.view_as(bn.bias)
                            bn_weights = bn_weights.view_as(bn.weight)
                            bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                            bn_running_var = bn_running_var.view_as(bn.running_var)

                            #Copy the data to model
                            
                            bn.bias.copy_(bn_biases)
                            bn.weight.copy_(bn_weights)
                            bn.running_mean.copy_(bn_running_mean)
                            bn.running_var.copy_(bn_running_var)
                    else:
                        #Number of biases
                        num_biases = conv.bias.numel()

                        #Load the weights
                        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                        ptr = ptr + num_biases

                        if not load_skip:
                            #reshape the loaded weights according to the dims of the model weights
                            conv_biases = conv_biases.view_as(conv.bias)

                            #Finally copy the data
                            conv.bias.copy_(conv_biases)
                    
                    num_weights = conv.weight.numel()

                    #Do the same as above for weights
                    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                    ptr = ptr + num_weights

                    if not load_skip:
                        conv_weights = conv_weights.view_as(conv.weight)
                        conv.weight.copy_(conv_weights)
