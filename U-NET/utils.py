# main_source https://github.com/kmaninis/OSVOS-PyTorch/blob/master/dataloaders/helpers.py

import os.path

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from torchvision.transforms import CenterCrop
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DAVIS2017
from config import (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        DEVICE,
        PAD_MIRRORING
)


# converts a torch.Tensor to np.array, removes the batch_dim, if array has more than one channel
# it transposes it from (CHANNELS, HEIGHT, WIDTH) to (HEIGHT, WIDTH, CHANNELS)
def tens2image(im):
    # removing batch size
    tmp = np.squeeze(im.numpy())
    # if greyscale
    if tmp.ndim == 2:
        return tmp
    else:
        # in order to perform np.subtract and to plot images, we have to transfer channel to the last dimension
        return tmp.transpose((1, 2, 0))


def overlay_mask(im, ma, color=np.array([255, 0, 0])/255.0):
    assert np.max(im) <= 255, "RGB channels' value cannot exceed 255"

    # float to bool means: each not-0 is set to 1, each 0 is kept 0
    ma = ma.astype(np.bool)
    im = im.astype(np.uint8)

    alpha = 0.5

    # fg = im*alpha + np.ones(im.shape)*(1-alpha) * np.array([23,23,197])/255.0
    # im (0, 1)
    # color (0, 1)
    # here we are creating a red transparent filter for the whole image that later will be
    # used only in the perimeter of the masks
    fg = im * alpha + np.ones(im.shape) * 255 * (1 - alpha) * color  # np.array([0,0,255])/255.0
    # Whiten background
    alpha = 1
    bg = im.copy()

    # create a simil_fg where fg is set to 0 where the mask is 0
    # substitute the values of the gt_image where mask != 0 with the values of the simil_fg where values are != 0

    # ma == 0 filters all the contour pixels of the mask
    # + np.ones(im[ma == 0].shape) * (1 - alpha) if alpha < 1 is used to whiten the background
    # if alpha=1 (default), + np.ones(im[ma == 0].shape) * (1 - alpha) is dropped

    bg[ma == 0] = im[ma == 0] * alpha + np.ones(im[ma == 0].shape) * 255 * (1 - alpha)

    bg[ma == 1] = fg[ma == 1]

    # cv2.findContours(image, mode, method)
    contours, _ = cv2.findContours(ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(image, contours, colourIdx, color, thickness)
    cv2.drawContours(bg, contours, -1, (0, 0, 0), 1)

    return bg


# rescales the image from (0, 1) to (0, 255)
def inv_normalize(im):
    assert np.max(im) <= 1 and np.min(im) >= 0, "Image is not scaled between 0 and 1"
    # im.min() = im.max() float32 (normalization not done channel-wise)

    return im * 255


def get_loaders(
    db_root_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = DAVIS2017(train=True, db_root_dir=db_root_dir, transform=train_transform,
                         pad_mirroring=PAD_MIRRORING)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = DAVIS2017(train=False, db_root_dir=db_root_dir, transform=val_transform,
                       pad_mirroring=PAD_MIRRORING)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # I have to shuffle otherwise the saved_images are retrieved from
        # the "bike-packing" class that is composed by very similar images
        # otherwise it would be unnecessary
        shuffle=True,
    )

    return train_loader, val_loader


# define train_loop
def train_loop(model, loader, optim, loss_fn):

    loop = tqdm(loader, position=0, leave=True)
    for image, mask in loader:
        # transferring data to cpu or gpu
        image = image.to(DEVICE)
        mask = mask.to(DEVICE).unsqueeze(dim=1)

        # feedforward
        out = model(image)
        loss = loss_fn(out, mask)

        # backpropagation
        loss.backward()
        optim.step()
        optim.zero_grad()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def dice_loss_accuracy(model,
                       val_loader,
                       device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for idx, (image, mask) in enumerate(val_loader):
            if idx==2:
                break
            image = image.to(device)
            mask = mask.to(device)
            mask_pred = torch.sigmoid(model(image))
            pred_correct = (mask_pred > 0.5).float()
            num_correct += (mask_pred == mask).sum()
            num_pixels += torch.numel(mask_pred)
            dice_score += (2 * (mask_pred * mask).sum()) / (
                    (mask_pred + mask).sum() + 1e-8
            )

        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
        )
        print(f"Dice score: {dice_score / len(val_loader)}")
        model.train()


def save_checkpoint(state, folder_path, filename="my_checkpoint.pth.tar"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print("=> Saving checkpoint")
    torch.save(state, os.path.join(folder_path,filename))


def save_images(model, loader, folder, epoch, device, num_batches, pad_mirroring):

    path = os.path.join(folder, f"epoch_{epoch+1}")
    if not os.path.exists(path):
        os.makedirs(path)

    model.eval()
    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            if idx < num_batches:
                images = images.to(device)
                outs = model(images)
                if pad_mirroring:
                    images = CenterCrop((IMAGE_HEIGHT, IMAGE_WIDTH))(images)
                # if the batch_size is > 1 we take just the first image/mask
                for i in range(images.shape[0]):

                    image = images[i]
                    mask = masks[i]
                    out = outs[i]

                    image = tens2image(image)
                    mask = tens2image(mask)
                    out = tens2image(out)

                    img_gt = overlay_mask(inv_normalize(image), mask)
                    img_out = overlay_mask(inv_normalize(image), out)

                    fig = plt.figure(figsize=(10, 7))
                    rows = 1
                    columns = 2

                    fig.add_subplot(rows, columns, 1)
                    plt.imshow(img_gt)
                    plt.axis('off')
                    plt.title("Input image and ground_truth mask")

                    fig.add_subplot(rows, columns, 2)
                    plt.imshow(img_out)
                    plt.axis('off')
                    plt.title("Input image and predicted_mask")

                    fig.savefig(f'{path}/image_{idx}.png')

    model.train()


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

