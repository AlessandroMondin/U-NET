import random
import numpy as np
import torch
import os
import warnings
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.utils import resize_image
from utils.bboxes_utils import rescale_bboxes, iou_width_height, coco_to_yolo, non_max_suppression as nms
from utils.plot_utils import plot_image, cells_to_bboxes, cells_to_bboxes2
import config
from check import plot_coco


class MS_COCO_2017(Dataset):
    """COCO 2017 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,
                 num_classes,
                 anchors,
                 root_directory=config.ROOT_DIR,
                 transform=None,
                 train=True,
                 S=(8, 16, 32),
                 adaptive_loader=False,
                 default_size=640,
                 bs=4,
                 ):
        """
        Parameters:
            train (bool): if true the os.path.join will lead to the train set, otherwise to the val set
            root_directory (path): path to the COCO2017 dataset
            transform: set of Albumentations transformations to be performed with A.Compose
        """
        self.nc = num_classes
        self.transform = transform
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.S = S
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
        self.adaptive_loader = adaptive_loader
        self.default_size = default_size
        self.root_directory = root_directory
        self.train = train

        if train:
            fname = 'images/train2017'
            annot_file = "coco_2017_train_csv.csv"
            # class instance because it's used in the __getitem__
            self.annot_folder = "coco_2017_train_txt"
        else:
            fname = 'images/val2017'
            annot_file = "coco_2017_val_csv.csv"
            # class instance because it's used in the __getitem__
            self.annot_folder = "coco_2017_val_txt"

        self.fname = fname

        self.annotations = pd.read_csv(os.path.join(root_directory, "annotations", annot_file), header=None)

        if adaptive_loader:
            self.annotations = self.adaptive_shape(self.annotations, bs)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_name = self.annotations.iloc[idx, 0]
        h = self.annotations.iloc[idx, 1]
        w = self.annotations.iloc[idx, 2]
        # img_name[:-4] to remove the .jpg or .png which are coco img formats
        label_path = os.path.join(os.path.join(config.ROOT_DIR, "annotations", self.annot_folder, img_name[:-4] + ".txt"))
        # to avoid an annoying "UserWarning: loadtxt: Empty input file"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            annotations = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()

        img = np.array(Image.open(os.path.join(config.ROOT_DIR, self.fname, img_name)).convert("RGB"))

        if self.adaptive_loader:
            bboxes = [ann[:-1] for ann in annotations]
            classes = [ann[-1] for ann in annotations]
            sh, sw = img.shape[0:2]
            img = resize_image(img, (w, h))
            bboxes = rescale_bboxes(bboxes, [sw, sh], [w, h])
            bboxes = [list(bboxes[i]) + [classes[i]] for i in range(len(bboxes))]

        if self.transform:
            augmentations = self.transform(image=img, bboxes=bboxes if self.adaptive_loader else annotations)
            img = augmentations["image"]
            bboxes = augmentations["bboxes"]

        return img, bboxes

    # this method modifies the target width and height of
    # the images by reshaping them so that the largest size of
    # a given image is set by its closest multiple to 640 (plus some
    # randomness and the other dimension is multiplied by the same scale
    # the purpose is multi_scale training by somehow preserving the
    # original ratio of images

    def adaptive_shape(self, annotations, batch_size):

        name = "train" if self.train else "val"
        path = os.path.join(
            self.root_directory, "annotations",
            "adaptive_ann_{}_bs_{}.csv".format(name, int(batch_size))
        )

        if os.path.isfile(path):
            print("==> Loading cached annotations for adaptive loader")
            parsed_annot = pd.read_csv(path, index_col=0)
        else:
            annotations.sort_values([1, 2], ascending=True, inplace=True)
            # IMPLEMENT POINT 2 OF WORD DOCUMENT
            for i in range(0, len(annotations), batch_size):
                size = [annotations.iloc[i, 2], annotations.iloc[i, 1]]  # [width, height]
                max_dim = max(size)
                max_idx = size.index(max_dim)
                size[~max_idx] += 32
                sz = random.randrange(int(self.default_size * 0.7), int(self.default_size * 1.3)) // 32 * 32
                size[~max_idx] = ((sz/size[max_idx])*(size[~max_idx]) // 32) * 32
                size[max_idx] = sz
                if i + batch_size <= len(annotations):
                    bs = batch_size
                else:
                    bs = len(annotations) - i
                for idx in range(bs):
                    annotations.iloc[i + idx, 2] = size[0]
                    annotations.iloc[i + idx, 1] = size[1]

                # sample annotation to avoid having pseudo-equal images in the same batch
                annotations.iloc[i:idx, :] = annotations.iloc[i:idx, :].sample(frac=1, axis=0)

            parsed_annot = pd.DataFrame(annotations)
            parsed_annot.to_csv(path)

        return parsed_annot

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class MS_COCO_2017_VALIDATION(Dataset):
    """COCO 2017 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,
                 num_classes,
                 anchors,
                 root_directory=config.ROOT_DIR,
                 transform=None,
                 train=True,
                 S=(8, 16, 32),
                 adaptive_loader=False,
                 default_size=640,
                 bs=4,
                 ):
        """
        Parameters:
            train (bool): if true the os.path.join will lead to the train set, otherwise to the val set
            root_directory (path): path to the COCO2017 dataset
            transform: set of Albumentation transformations to be performed with A.Compose
        """
        self.nc = num_classes
        self.transform = transform
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.S = S
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
        self.adaptive_loader = adaptive_loader
        self.default_size = default_size
        self.root_directory = root_directory
        self.train = train

        if train:
            fname = 'images/train2017'
            annot_file = "coco_2017_train_csv.csv"
            # class instance because it's used in the __getitem__
            self.annot_folder = "coco_2017_train_txt"
        else:
            fname = 'images/val2017'
            annot_file = "coco_2017_val_csv.csv"
            # class instance because it's used in the __getitem__
            self.annot_folder = "coco_2017_val_txt"

        self.fname = fname

        self.annotations = pd.read_csv(os.path.join(root_directory, "annotations", annot_file), header=None)

        if adaptive_loader:
            self.annotations = self.adaptive_shape(self.annotations, bs)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_name = self.annotations.iloc[idx, 0]
        print(img_name)

        tg_height = self.annotations.iloc[idx, 1] if self.adaptive_loader else 640
        tg_width = self.annotations.iloc[idx, 2] if self.adaptive_loader else 640
        # print(f'image_name: {img_name}, idx: {idx}, tg_height: {tg_height}, tg_width: {tg_width}')
        # img_name[:-4] to remove the .jpg or .png which are coco img formats
        label_path = os.path.join(os.path.join(config.ROOT_DIR, "annotations", self.annot_folder, img_name[:-4] + ".txt"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            annotations = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()

        bboxes = [ann[:-1] for ann in annotations]
        classes = [ann[-1] for ann in annotations]

        img = np.array(Image.open(os.path.join(config.ROOT_DIR, self.fname, img_name)).convert("RGB"))

        if self.adaptive_loader:
            sh, sw = img.shape[0:2]
            # print(f'starting_height: {sh}, starting_width: {sw}')
            bboxes = rescale_bboxes(bboxes, [sw, sh], [tg_width, tg_height])
            img = resize_image(img, (tg_width, tg_height))

        # annot_bboxes = [[classes[i]-1]+[1]+list(bboxes[i]) for i in range(len(bboxes))]
        # plot_coco(img, annot_bboxes)

        if self.transform:
            augmentations = self.transform(image=img, bboxes=bboxes)
            img = augmentations["image"]
            bboxes = augmentations["bboxes"]


        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # 6 because (p_o, x, y, w, h, class)
        # targets is a list of len 3 and targets[0] has shape (3, 13, 13 ,6)
        # ?where is batch_size?
        targets = [torch.zeros((self.num_anchors // 3, int(img.shape[1]/S), int(img.shape[2]/S), 6)) for S in self.S]
        for idx, box in enumerate(bboxes):
            class_label = classes[idx] - 1  # classes in coco start from 1
            box = coco_to_yolo(box, image_w=tg_width, image_h=tg_height)
            # this iou() computer iou just by comparing widths and heights
            # torch.tensor(box[2:4] -> shape (2,) - self.anchors shape -> (9,2)
            # iou_anchors --> tensor of shape (9,)
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors/torch.tensor([640, 640]))
            # sorting anchors from the one with best iou with gt_box
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, = box

            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:
                # i.e. if the best anchor idx is 8, num_anchors_per_scale
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

                scale_y = targets[scale_idx].shape[1]
                scale_x = targets[scale_idx].shape[2]
                # S = self.S[scale_idx]
                # scale_y = int(img.shape[1]/S)
                # scale_x = int(img.shape[2]/S)

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

        return img, tuple(targets)

    # this method modifies the target width and height of
    # the images by reshaping them so that the largest size of
    # a given image is set by its closest multiple to 640 (plus some
    # randomness and the other dimension is multiplied by the same scale
    # the purpose is multi_scale training by somehow preserving the
    # original ratio of images

    def adaptive_shape(self, annotations, batch_size):

        name = "train" if self.train else "val"
        path = os.path.join(
            self.root_directory, "annotations",
            "adaptive_ann_{}_bs_{}.csv".format(name, int(batch_size))
        )

        if os.path.isfile(path):
            print("==> Loading cached annotations for adaptive loader")
            parsed_annot = pd.read_csv(path, index_col=0)
        else:
            annotations.sort_values([1, 2], ascending=True, inplace=True)
            # IMPLEMENT POINT 2 OF WORD DOCUMENT
            for i in range(0, len(annotations), batch_size):
                size = [annotations.iloc[i, 2], annotations.iloc[i, 1]]  # [width, height]
                max_dim = max(size)
                max_idx = size.index(max_dim)
                size[~max_idx] += 32
                sz = random.randrange(int(self.default_size * 0.7), int(self.default_size * 1.3)) // 32 * 32
                size[~max_idx] = ((sz/size[max_idx])*(size[~max_idx]) // 32) * 32
                size[max_idx] = sz
                if i + batch_size <= len(annotations):
                    bs = batch_size
                else:
                    bs = len(annotations) - i
                for idx in range(bs):
                    annotations.iloc[i + idx, 2] = size[0]
                    annotations.iloc[i + idx, 1] = size[1]

                # sample annotation to avoid having pseudo-equal images in the same batch
                annotations.iloc[i:idx, :] = annotations.iloc[i:idx, :].sample(frac=1, axis=0)

            parsed_annot = pd.DataFrame(annotations)
            parsed_annot.to_csv(path)

        return parsed_annot

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == "__main__":

    S = [8, 16, 32]

    anchors = config.ANCHORS

    transform = config.VAL_TRANSFORM

    dataset = MS_COCO_2017_VALIDATION(num_classes=len(config.COCO_LABELS), anchors=config.ANCHORS,
                                      root_directory=config.ROOT_DIR, transform=config.ADAPTIVE_VAL_TRANSFORM,
                                      train=True, S=S, adaptive_loader=True, default_size=640, bs=64)

    anchors = torch.tensor(anchors)
    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=False)

    for x, y in loader:

        boxes = []
        boxes2 = cells_to_bboxes2(y, anchors, S)
        """
        for i in range(y[0].shape[1]):
            anchor = anchors[i]

            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]"""

        boxes = nms(boxes2, iou_threshold=1, threshold=0.7, box_format="midpoint")

        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes, cell_to_bboxes1=False)