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
ROOT_DIR = "SageMaker/DAVIS_480/"
EPOCHS = 30
LEARNING_RATE = 1e-4
PAD_MIRRORING = 92
SAVE_MODEL_PATH = "SageMaker/saved_checkpoint/mir_not_blur"
SAVE_IMAGES_PATH = "SageMaker/saved_images/mir_not_blur"
CHECKPOINT = "SageMaker/saved_checkpoint/mir_not_blur/checkpoint_epoch_10.pth.tar"
#CHECKPOINT = None


train_transform = A.Compose(
    [
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Rotate(limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
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


