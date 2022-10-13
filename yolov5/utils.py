import os
import torch
import torch.nn as nn
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import config


# Tensor.element_size() → int
# Returns the size in bytes of an individual element.
def save_model(model, folder_path, file_name):
    ckpt = {}
    ckpt["model"] = model
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print("Saving Model...")
    torch.save(ckpt, os.path.join(folder_path, file_name))


def check_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def strip_model(model):
    model.half()
    for p in model.parameters():
        p.requires_grid = False


def export_onnx(model):
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    input_names = ["actual_input"]
    output_names = ["output"]
    torch.onnx.export(model,
                      dummy_input,
                      "netron_onnx_files/yolov5m_mine.onnx",
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      export_params=True,
                      opset_version=11
                      )


# found here: https://gist.github.com/cbernecker/1ac2f9d45f28b6a4902ba651e3d4fa91#file-coco_to_yolo-py
def coco_to_yolo(bbox, image_w=640, image_h=640):
    x1, y1, w, h = bbox
    return [((2*x1 + w)/(2*image_w)), ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]


# rescales bboxes from an image_size to another image_size
def rescale_bboxes(bboxes, starting_size, ending_size):
    sw, sh = starting_size
    ew, eh = ending_size
    new_boxes = []
    for bbox in bboxes:
        x = bbox[0] * ew/sw
        y = bbox[1] * eh/sh
        w = bbox[2] * ew/sw
        h = bbox[3] * eh/sh
        new_boxes.append([x, y, w, h])
    return new_boxes


# NOT USED
def reshape_bbox(bbox, ratio_w, ratio_h):
    """
    Parameters:
        bbox (lists): bbox (x0,y0,w,h)
        ratio_w (float): ratio--> width_after_transform/width_before_transform
        ratio_h (float): ratio--> height_after_transform/height_before_transform
    Returns:
        list of reshaped bounding boxes
    """
    bbox = list(bbox)
    bbox[0] = int(ratio_w * bbox[0])
    bbox[1] = int(ratio_h * bbox[1])
    bbox[2] = int(ratio_w * bbox[2])
    bbox[3] = int(ratio_h * bbox[3])

    return bbox


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
    bboxes = list(map(lambda box: reshape_bbox(box, ratio_w=new_w/w, ratio_h=new_h/h), bboxes))
    # ns are the height,width that the new image will have
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    return torch.squeeze(nn.functional.interpolate(img, size=ns, mode="bilinear", align_corners=False)), bboxes


def multi_scale(img, target_shape, max_stride):
    # to make it work with collate_fn of the loader
    if isinstance(img, tuple):
        img = torch.stack(list(img), dim=0)
    # returns a random number between target_shape*0.5 e target_shape*1.5+max_stride, applies an integer
    # division by max stride and multiplies again for max_stride
    # in other words it returns a number between those two interval divisible by 32
    sz = random.randrange(target_shape * 0.5, target_shape + max_stride) // max_stride * max_stride
    # sf is the ratio between the random number and the max between height and width
    sf = sz / max(img.shape[2:])
    h, w = img.shape[2:]
    # 1) regarding the larger dimension (height or width) it will become the closest divisible by 32 of
    # larger_dimension*sz
    # 2) regarding the smaller dimension (height or width) it will become the closest divisible by 32 of
    # smaller_dimension*sf (random_number_divisible_by_32_within_range/larger_dimension)
    # math.ceil is the opposite of floor, it rounds the floats to the next ints
    ns = [math.ceil(i * sf / max_stride) * max_stride for i in [h, w]]
    # ns are the height,width that the new image will have
    imgs = nn.functional.interpolate(img, size=ns, mode="bilinear", align_corners=False)
    return imgs, imgs.shape[2:4]


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def iou_width_height(gt_box, anchors):
    """
    Parameters:
        gt_box (tensor): width and height of the ground truth box
        anchors (tensor): lists of anchors containing width and height
    Returns:
        tensor: Intersection over union between the gt_box and each of the n-anchors
    """
    # boxes 1 (gt_box): shape (2,)
    # boxes 2 (anchors): shape (9,2)
    # intersection shape: (9,)
    intersection = torch.min(gt_box[..., 0], anchors[..., 0]) * torch.min(
        gt_box[..., 1], anchors[..., 1]
    )
    union = (
        gt_box[..., 0] * gt_box[..., 1] + anchors[..., 0] * anchors[..., 1] - intersection
    )
    # intersection/union shape (9,)
    return intersection / union


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()

