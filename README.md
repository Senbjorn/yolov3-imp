# YOLOv3 implementation
This is an implementation of YOLOv3. First part is based on this series of [articles](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/). All code is reviewed and reimplemented. Also, some improvements were introduced. I avoided execeptions as they could be replaced with `if` statements and used `detach()` or `torch.no_grad()` instead of `.data`. Functions for computing IOU and bounding box transformations are restructured.

If you have problems with viewing .ipynb, go to [nbviewer](https://nbviewer.jupyter.org/github/Senbjorn/yolov3-imp/blob/main/YOLOv3.ipynb).
