# main source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/train.py
import os.path

from model import UNET
from utils import (
    get_loaders,
    dice_loss_accuracy,
    save_checkpoint,
    save_images,
    train_loop
)
from config import (
    ROOT_DIR,
    train_transform,
    val_transforms,
    DEVICE,
    LOSS_FN,
    LEARNING_RATE,
    EPOCHS,
    PAD_MIRRORING,
    FOLDER_PATH
)
from torch.optim import Adam

# we are going to train and compare 3 models:
# 1) input and target 388x388 --> apply mirroring to input to widen it to 572
# 2) input and target 572x572 --> apply center_crop on the mask during forward()
# 2) input and target 572x572, model with padding.


def main():

    model = UNET(in_channels=3, expansion=64, exit_channels=1)
    optim = Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(db_root_dir=ROOT_DIR, batch_size=2, train_transform=train_transform,
                                           val_transform=val_transforms, num_workers=1)
    for epoch in range(EPOCHS):

        # train_loop(model=model, loader=train_loader, epochs=10, loss_fn=LOSS_FN, optim=optim)

        dice_loss_accuracy(model, val_loader, device=DEVICE)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
        }
        save_checkpoint(checkpoint, folder_path=FOLDER_PATH,
                        filename=f"checkpoint_epoch_{epoch+1}.pth.tar")

        save_images(model=model, loader=val_loader, folder="saved_images",
                    epoch=epoch, device=DEVICE, num_batches=5, pad_mirroring=PAD_MIRRORING)


if __name__ == "__main__":
    main()










