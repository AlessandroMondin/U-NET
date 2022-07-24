from PIL import Image
import numpy as np
from model import UNET
from utils import (
    save_checkpoint,
    load_model_checkpoint,
    save_images,
    predict_image
)
from config import (
    val_transforms,
    DEVICE,
    PAD_MIRRORING,
    CHECKPOINT,
    IMAGE_PATH
)

if __name__ == "__main__":
    image = np.array(Image.open(IMAGE_PATH).convert("RGB"), dtype=np.float32)
    model = UNET(3, 64, 1, padding=0, downhill=4).to(DEVICE)
    load_model_checkpoint(CHECKPOINT, model)
    predict_image(model, image, val_transforms, folder="SageMaker/saved_predictions",
                  image_title="leao", pad_mirroring = PAD_MIRRORING)