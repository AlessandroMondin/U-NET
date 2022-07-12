import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from model import UNET

"""
Here below the parameter:
Notice that if you want to change IMAGE_HEIGHT and IMAGE_WIDTH you have to modify
accordingly MASK_HEIGHT and MASK_WIDTH by running the 4 rows hashed below
"""

IMAGE_HEIGHT = 388
IMAGE_WIDTH = 388
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_FN = torch.nn.BCEWithLogitsLoss()
ROOT_DIR = "/Users/alessandro/Desktop/ML/DL_DATASETS/DAVIS_480/"
EPOCHS = 10
LEARNING_RATE = 3e-4
PAD_MIRRORING = 92
FOLDER_PATH = "saved_checkpoint/mirroring_input"


train_transform = A.Compose(
    [
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Rotate(limit=45, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Blur(blur_limit=7, p=0.3),
        A.Normalize(
            mean=(0, 0, 0),
            std=(1, 1, 1),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(
            mean=(0, 0, 0),
            std=(1, 1, 1),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


