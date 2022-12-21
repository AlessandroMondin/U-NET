# main source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/train.py
import os.path
import torch
from model import UNET
from utils import (
    get_loaders,
    evalution_metrics,
    save_checkpoint,
    load_model_checkpoint,
    load_optim_checkpoint,
    save_images,
    train_loop
)
from config import (
    ROOT_DIR,
    train_transform,
    val_transforms,
    DEVICE,
    LEARNING_RATE,
    EPOCHS,
    PAD_MIRRORING,
    CHECKPOINT,
    SAVE_MODEL_PATH,
    SAVE_IMAGES_PATH
)
from torch.optim import Adam

# we are going to train and compare 3 models:
# 1) input and target 388x388 --> apply mirroring to input to widen it to 572
# 2) input and target 572x572 --> apply center_crop on the mask during forward()
# 2) input and target 572x572, model with padding.


def main():
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    model = UNET(3, 64, 1, padding=0, downhill=4).to(DEVICE)
    optim = Adam(model.parameters(), lr=LEARNING_RATE)
    
    if CHECKPOINT:
        load_model_checkpoint(CHECKPOINT, model)
        load_optim_checkpoint(CHECKPOINT, optim)
    
    train_loader, val_loader = get_loaders(db_root_dir=ROOT_DIR, batch_size=8, train_transform=train_transform,
                                           val_transform=val_transforms, num_workers=4)
    for epoch in range(17, EPOCHS):
        
        print(f"Training epoch {epoch+1}/{EPOCHS}")

        train_loop(model=model, loader=train_loader, loss_fn=loss_fn, optim=optim, scaler=scaler, pos_weight=False)
        
        print("Computing valuation metrics on val_loader...")
        
        evalution_metrics(model, val_loader, loss_fn, device=DEVICE)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
        }
        
        save_checkpoint(checkpoint, folder_path=SAVE_MODEL_PATH,
                        filename=f"checkpoint_epoch_{epoch+1}.pth.tar")

        save_images(model=model, loader=val_loader, folder=SAVE_IMAGES_PATH,
                    epoch=epoch, device=DEVICE, num_images=10, pad_mirroring=PAD_MIRRORING)
        


if __name__ == "__main__":
    main()
