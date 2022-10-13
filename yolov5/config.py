from pathlib import Path
import os

import albumentations as A
import torch.cuda
from albumentations.pytorch import ToTensorV2
import cv2


parent_dir = Path(__file__).parent.parent

if "/Users/alessandro" in str(parent_dir):
    ROOT_DIR = os.path.join(parent_dir, "Desktop", "ML", "DL_DATASETS", "COCO")
else:
    ROOT_DIR = os.path.join(parent_dir, "coco")
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
SAVE_MODEL_PATH = os.path.join(os.getcwd(), "SAVED_CHECKPOINT")
SAVE_IMAGES_PATH = "SageMaker/check/check"
IMAGE_SIZE = 640


CONF_THRESHOLD = 0.05  # ALADDIN set it to 0.05? Why so low?
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45

CHECKPOINT = None
#CHECKPOINT = os.path.join(os.getcwd(), "SAVED_CHECKPOINT", "checkpoint_epoch_1.pth.tar")

# triple check what anchors REALLY are
ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32#
]


TRAIN_TRANSFORMS = A.Compose(
    [
        A.Resize(width=640, height=640, interpolation=cv2.INTER_LINEAR),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.4),
        A.OneOf(
            [
             # when rotating an image, bboxes transformation ain't precise, you can either choose to
             # create rotated bboxes slightly smaller with rotate_method="ellipse" or
             # than gt_bboxes or quite wider with rotate_method="ellipse"largest_box"
             A.ShiftScaleRotate(
                rotate_limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT, rotate_method="ellipse",
                scale_limit=0.0, shift_limit=0
             ),
             A.ShiftScaleRotate(
                rotate_limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT, rotate_method="largest_box",
                scale_limit=0.0, shift_limit=0)
            ],
            p=0.7
        ),

        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="coco", min_visibility=0.4, label_fields=[],),
)


VAL_TRANSFORM = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="coco", min_visibility=0.4, label_fields=[]),
)

ADAPTIVE_TRAIN_TRANSFORM = A.Compose(
    # removing the A.Resize from the augmentations
    TRAIN_TRANSFORMS[1:],
    bbox_params=A.BboxParams(format="coco", min_visibility=0.4, label_fields=[])
)

ADAPTIVE_VAL_TRANSFORM = A.Compose(
    # removing the A.Resize from the augmentations
    VAL_TRANSFORM[1:],
    bbox_params=A.BboxParams(format="coco", min_visibility=0.4, label_fields=[])
)

COCO_LABELS=['person',
             'bicycle',
             'car',
             'motorcycle',
             'airplane',
             'bus',
             'train',
             'truck',
             'boat',
             'traffic light',
             'fire hydrant',
             'street sign',
             'stop sign',
             'parking meter',
             'bench',
             'bird',
             'cat',
             'dog',
             'horse',
             'sheep',
             'cow',
             'elephant',
             'bear',
             'zebra',
             'giraffe',
             'hat',
             'backpack',
             'umbrella',
             'shoe',
             'eye glasses',
             'handbag',
             'tie',
             'suitcase',
             'frisbee',
             'skis',
             'snowboard',
             'sports ball',
             'kite',
             'baseball bat',
             'baseball glove',
             'skateboard',
             'surfboard',
             'tennis racket',
             'bottle',
             'plate',
             'wine glass',
             'cup',
             'fork',
             'knife',
             'spoon',
             'bowl',
             'banana',
             'apple',
             'sandwich',
             'orange',
             'broccoli',
             'carrot',
             'hot dog',
             'pizza',
             'donut',
             'cake',
             'chair',
             'couch',
             'potted plant',
             'bed',
             'mirror',
             'dining table',
             'window',
             'desk',
             'toilet',
             'door',
             'tv',
             'laptop',
             'mouse',
             'remote',
             'keyboard',
             'cell phone',
             'microwave',
             'oven',
             'toaster',
             'sink',
             'refrigerator',
             'blender',
             'book',
             'clock',
             'vase',
             'scissors',
             'teddy bear',
             'hair drier',
             'toothbrush',
             'hairbrush',
            ]
