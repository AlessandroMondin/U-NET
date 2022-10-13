import torch
import torch.nn as nn
import random
import math
from bboxes_utils import rescale_bboxes


# NOT USED
def multi_shape_one_img(img, target_shape, max_stride, bboxes):
    # returns a random number between target_shape*0.5 e target_shape*1.5+max_stride, applies an integer
    # division by max stride and multiplies again for max_stride
    # in other words it returns a number between those two interval divisible by 32
    sz = random.randrange(target_shape * 0.5, target_shape + max_stride) // max_stride * max_stride
    # sf is the ratio between the random number and the max between height and width
    sf = sz / max(img.shape[2:])
    h, w = img.shape[2:] if len(img.shape) == 4 else img.shape[1:]
    # 1) regarding the larger dimension (height or width) it will become the closest divisible by 32 of
    # larger_dimension*sz
    # 2) regarding the smaller dimension (height or width) it will become the closest divisible by 32 of
    # smaller_dimension*sf (random_number_divisible_by_32_within_range/larger_dimension)
    # math.ceil is the opposite of floor, it rounds the floats to the next ints
    ns = [math.ceil(i * sf / max_stride) * max_stride for i in [h, w]]
    new_h, new_w = ns
    bboxes = list(map(lambda box: rescale_bboxes(box, ratio_w=new_w/w, ratio_h=new_h/h), bboxes))
    # ns are the height,width that the new image will have
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    return torch.squeeze(nn.functional.interpolate(img, size=ns, mode="bilinear", align_corners=False)), bboxes

# NOT USED
def my_interpolation(img, thresh_w, thresh_h, max_stride=32):
    if isinstance(img, torch.Tensor):
        h, w = img.shape[2:]
    else:
        h, w = img
    if h > thresh_h:
        h = (random.randrange(int(h*0.7), h)//32)*32
    else:
        h = (h // 32) * 32
    if w > thresh_w:
        w = (random.randrange(int(w*0.7), w)//32)*32
    else:
        w = (w//32)*32

    return nn.functional.interpolate(img, size=[h,w], mode="bilinear", align_corners=False)