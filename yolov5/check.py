# checking that all the images & labels are being loaded without triggering errors

from dataset import MS_COCO_2017, MS_COCO_2017_VALIDATION
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
from time import sleep

if __name__ == "__main__":
    S = [8, 16, 32]
    training = [True, False]
    for train in training:

        transform = config.TRAIN_TRANSFORMS if train else config.VAL_TRANSFORM

        dataset_train = MS_COCO_2017(num_classes=len(config.COCO_LABELS), anchors=config.ANCHORS, root_directory=config.ROOT_DIR,
                                     transform=config.VAL_TRANSFORM, train=train, S=S)

        dataset_val = MS_COCO_2017_VALIDATION(num_classes=len(config.COCO_LABELS), anchors=config.ANCHORS,
                                              root_directory=config.ROOT_DIR, transform=config.VAL_TRANSFORM,
                                              train=train, S=S)

        loader_train = DataLoader(dataset_train, batch_size=8, collate_fn=dataset_train.collate_fn, num_workers=0, shuffle=False)
        loader_val = DataLoader(dataset=dataset_val, batch_size=8, shuffle=True)

        train_loop = tqdm(loader_train)

        for idx, (images, labels, _) in enumerate(train_loop):
            if idx > 100:
                break
        print("Success on training dataset with train_transform == {}".format(train))
        sleep(1)
        val_loop = tqdm(loader_val)
        for images, labels in val_loop:
            a = 1
        print("Success on val dataset with train_transform == {}".format(train))