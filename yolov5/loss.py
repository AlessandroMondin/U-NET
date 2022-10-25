import time
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.training_utils import multi_scale
from utils.bboxes_utils import (
    iou_width_height,
    coco_to_yolo,
    rescale_bboxes,
    intersection_over_union,
    non_max_suppression as nms
)
from utils.plot_utils import cells_to_bboxes, plot_image, cells_to_bboxes2
import config
from model import YOLOV5m
from dataset import MS_COCO_2017
import torch.nn.functional as F


# found here: https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class YOLO_LOSS:
    def __init__(self, model, adaptive_loader, save_logs=False):

        self.adaptive_loader = adaptive_loader
        self.mse = nn.MSELoss()
        self.BCE_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.CLS_PW))
        self.BCE_obs = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.OBJ_PW))

        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 0.5
        self.lambda_obj = 1
        self.lambda_box = 0.05

        anchors = getattr(model.head, "anchors").clone().detach()
        self.anchors = torch.cat((anchors[0], anchors[1], anchors[2]), dim=0)

        self.na = self.anchors.shape[0]
        self.num_anchors_per_scale = self.na // 3
        self.S = getattr(model.head, "stride")
        self.ignore_iou_thresh = 0.5
        self.ph = None  # this variable is used in the build_targets method, defined here for readability.
        self.pw = None  # this variable is used in the build_targets method, defined here for readability.
        self.save_logs = save_logs

        if self.save_logs:
            with open(os.path.join("train_eval_metrics", "training_logs", "loss.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "batch_idx", "box_loss", "object_loss", "class_loss"])
                print("--------------------------------------------------------------------------------------")
                print("Created loss.csv in /train_eval_logs/training_logs, it will be updated during training..")
                print("--------------------------------------------------------------------------------------")
                f.close()

    def __call__(self, preds, targets, pred_size, batch_idx=None, epoch=None):
        self.batch_idx = batch_idx
        self.epoch = epoch

        # list of lists --> [pred[0].height, pred[0].width, pred[1].height... etc]

        targets = [self.build_targets(preds, bboxes, pred_size) for bboxes in targets]

        t1 = torch.stack([target[0] for target in targets], dim=0).to(config.DEVICE)
        t2 = torch.stack([target[1] for target in targets], dim=0).to(config.DEVICE)
        t3 = torch.stack([target[2] for target in targets], dim=0).to(config.DEVICE)

        anchors = self.anchors.reshape(3, 3, 2).to(config.DEVICE)

        if self.save_logs:
            l1, logs1 = self.compute_loss(preds[0], t1, anchors=anchors[0])
            l2, logs2 = self.compute_loss(preds[1], t2, anchors=anchors[1])
            l3, logs3 = self.compute_loss(preds[2], t3, anchors=anchors[2])
            loss = l1 + l2 + l3

            freq = 10
            if self.batch_idx % freq == 0:
                log_losses = torch.mean(torch.cat([logs1, logs2, logs3], dim=0), dim=0)
                with open(os.path.join("train_eval_metrics", "training_logs", "loss.csv"), "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.epoch, self.batch_idx, log_losses[0].item(), log_losses[1].item(),
                                     log_losses[2].item(), log_losses[3].item()])

                    f.close()

        else:
            loss = (
                self.compute_loss(preds[0], t1, anchors=anchors[0])
                + self.compute_loss(preds[1], t2, anchors=anchors[1])
                + self.compute_loss(preds[2], t3, anchors=anchors[2])
            )

        return loss

    def build_targets(self, input_tensor, bboxes, pred_size):
        check_loss = False
        if check_loss:
            ph = pred_size[0]
            pw = pred_size[1]
        else:
            pw, ph = input_tensor.shape[3], input_tensor.shape[2]
        # for loop long ain't elegant :-(
        
        if check_loss:
            targets = [
                torch.zeros((self.num_anchors_per_scale, input_tensor[i].shape[2], input_tensor[i].shape[3], 6))
                for i in range(len(self.S))
            ]
        else:
            targets = [torch.zeros((self.num_anchors_per_scale, int(input_tensor.shape[2]/S),
                                    int(input_tensor.shape[3]/S), 6)) for S in self.S]

        classes = [box[-1] for box in bboxes]
        bboxes = [box[:-1] for box in bboxes]

        if not self.adaptive_loader:
            if check_loss:
                bboxes = rescale_bboxes(bboxes, starting_size=(640, 640), ending_size=(pw, ph))
            else:
                bboxes = rescale_bboxes(bboxes, starting_size=(640, 640), ending_size=(pw, ph))

        for idx, box in enumerate(bboxes):
            class_label = classes[idx] - 1  # classes in coco start from 1
            box = coco_to_yolo(box, pw, ph)

            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors/torch.tensor([640, 640]).to(config.DEVICE))
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            x, y, width, height, = box
            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:
                # i.e if the best anchor idx is 8, num_anchors_per_scale
                # we know that 8//3 = 2 --> the best scale_idx is 2 -->
                # best_anchor belongs to last scale (52,52)
                # scale_idx will be used to slice the variable "targets"
                # another pov: scale_idx searches the best scale of anchors
                scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode="floor")
                # print(scale_idx)
                # anchor_on_scale searches the idx of the best anchor in a given scale
                # found via index in the line below
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                # slice anchors based on the idx of the best scales of anchors
                if check_loss:
                    scale_y = input_tensor[int(scale_idx)].shape[2]
                    scale_x = input_tensor[int(scale_idx)].shape[3]
                else:
                    S = self.S[scale_idx]
                    scale_y = int(input_tensor.shape[2] / S)
                    scale_x = int(input_tensor.shape[3] / S)

                # S = self.S[int(scale_idx)]
                # another problem: in the labels the coordinates of the objects are set
                # with respect to the whole image, while we need them wrt the corresponding (?) cell
                # next line idk how --> i tells which y cell, j which x cell
                # i.e x = 0.5, S = 13 --> int(S * x) = 6 --> 6th cell
                i, j = int(scale_y * y), int(scale_x * x)  # which cell
                # targets[scale_idx] --> shape (3, 13, 13, 6) best group of anchors
                # targets[scale_idx][anchor_on_scale] --> shape (13,13,6)
                # i and j are needed to slice to the right cell
                # 0 is the idx corresponding to p_o
                # I guess [anchor_on_scale, i, j, 0] equals to [anchor_on_scale][i][j][0]
                # check that the anchor hasn't been already taken by another object (rare)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                # if not anchor_taken == if anchor_taken is still == 0 cause in the following
                # lines will be set to one
                # if not has_anchor[scale_idx] --> if this scale has not been already taken
                # by another anchor which were ordered in descending order by iou, hence
                # the previous ones are better
                if not anchor_taken and not has_anchor[scale_idx]:
                    # here below we are going to populate all the
                    # 6 elements of targets[scale_idx][anchor_on_scale, i, j]
                    # setting p_o of the chosen cell = 1 since there is an object there
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    # setting the values of the coordinates x, y
                    # i.e (6.5 - 6) = 0.5 --> x_coord is in the middle of this particular cell
                    # both are between [0,1]
                    x_cell, y_cell = scale_x * x - j, scale_y * y - i  # both between [0,1]
                    # width = 0.5 would be 0.5 of the entire image
                    # and as for x_cell we need the measure w.r.t the cell
                    # i.e S=13, width = 0.5 --> 6.5
                    width_cell, height_cell = (
                        width * scale_x,
                        height * scale_y,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                # not understood

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return targets

    # TRAINING_LOSS
    def compute_loss(self, preds, targets, anchors):

        # originally anchors have shape (3,2) --> 3 set of anchors of width and height
        bs = preds.shape[0]
        anchors = anchors.reshape(1, 3, 1, 1, 2)

        pxy = (preds[..., 1:3].sigmoid() * 2) - 0.5
        pwh = ((preds[..., 3:5].sigmoid() * 2) ** 2) * anchors
        pbox = torch.cat((pxy, pwh), dim=-1)
        tbox = targets[..., 1:5]

        # ======================= #
        #   FOR OBJECTNESS SCORE    #
        # ======================= #

        lobj = self.BCE_obs(preds[..., 0:1], targets[..., 0:1])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        iou = intersection_over_union(pbox, tbox, GIoU=True).squeeze()  # iou(prediction, target)
        lbox = (1.0 - iou).mean()  # iou loss

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        # NB: my targets[...,5:6]) is a vector of size bs, 1,
        # ultralytics targets[...,5:6]) is a matrix of shape bs, num_classes
        lcsl = self.BCE_cls(preds[..., 5:6], targets[..., 5:6])  # BCE

        return (
            (self.lambda_box * lbox
             + self.lambda_obj * lobj
             + self.lambda_class * lcsl) * bs,

            torch.unsqueeze(
                torch.stack([
                    self.lambda_box * lbox,
                    self.lambda_obj * lobj,
                    self.lambda_class * lcsl
                ]), dim=0
            )

            if self.save_logs else None
        )


if __name__ == "__main__":
    check_loss = False
    batch_size = 8
    image_height = 640
    image_width = 640
    nc = 91
    S = [8, 16, 32]

    anchors = config.ANCHORS
    first_out = 48

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=anchors,
                    ch=(first_out*4, first_out*8, first_out*16), inference=False).to(config.DEVICE)

    dataset = MS_COCO_2017(num_classes=len(config.COCO_LABELS), anchors=config.ANCHORS,
                           root_directory=config.ROOT_DIR, transform=config.ADAPTIVE_VAL_TRANSFORM,
                           train=True, S=S, adaptive_loader=True, default_size=640, bs=8)

    anchors = torch.tensor(anchors)

    yolo_loss = YOLO_LOSS(model, adaptive_loader=dataset.adaptive_loader)

    loader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn)

    if check_loss:
        for images, bboxes in loader:
            images = torch.stack(images, dim=0)
            if not dataset.adaptive_loader:
                images = multi_scale(images, target_shape=640, max_stride=32).to(config.DEVICE)

            preds = model(images)
            start = time.time()
            loss = yolo_loss(preds, bboxes, pred_size=images.shape[2:4])
            print(loss)
            end = time.time()
            print(end-start)

    else:
        for images, bboxes in loader:
            images = torch.stack(images, dim=0)
            if not dataset.adaptive_loader:
                images = multi_scale(images, target_shape=640, max_stride=32).to(config.DEVICE)

            images = torch.unsqueeze(images[0], dim=0)  # keep just the first img but preserving bs
            bboxes = bboxes[0]
            targets = yolo_loss.build_targets(images, bboxes, images[0].shape[2:4])
            boxes = []
            S = [8, 16, 32]
            targets = [torch.unsqueeze(target, dim=0) for target in targets]

            boxes = cells_to_bboxes2(targets, anchors, S)
            boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")

            plot_image(images[0].permute(1, 2, 0).to("cpu"), boxes, cell_to_bboxes1=False)
