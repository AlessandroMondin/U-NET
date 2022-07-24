# found in https://github.com/kmaninis/OSVOS-PyTorch/blob/master/dataloaders/davis_2016.py

from torchvision.transforms import Pad
import os
from PIL import Image
from utils import *
from torch.utils.data import Dataset
from config import (
    train_transform,
    ROOT_DIR,
    PAD_MIRRORING,
    IMAGE_HEIGHT, IMAGE_WIDTH
)


class DAVIS2017(Dataset):
    """DAVIS 2017 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 db_root_dir=ROOT_DIR,
                 transform=None,
                 seq_name=None,
                 pad_mirroring=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        Parameters:
            train (bool): if true the os.path.join will lead to the train set, otherwise to the val set
            inputRes (tuple): image size after reshape (HEIGHT, WIDTH)
            db_root_dir (path): path to the DAVIS2017 dataset
            transform: set of Albumentation transformations to be performed with A.Compose
            meanval (tuple): set of magic weights used for normalization (np.subtract(im, meanval))
            seq_name (str): name of a class: i.e. if "bear" one im of "bear" class will be retrieved
        """
        self.train = train
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.seq_name = seq_name
        self.pad_mirroring = pad_mirroring

        if self.train==1:
            fname = 'train'
        elif self.train==0:
            fname = 'val'
        else:
            fname = "test-dev"

        if self.seq_name is None:

            # Initialize the original DAVIS splits for training the parent network
            # even though we could avoid using the txt files, we might have to use them
            # due to consistency: maybe some sub-folders shouldn't be included and we know which
            # to consider in the .txt file only
            with open(os.path.join(db_root_dir, "ImageSets/2017", fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    # why sort? And are we using np.sort cause we need the data-structure to be np.array
                    # instead of a list? Maybe it's faster
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                    # why using lambda? map applies a given function to each item of an iterable. Apparently
                    # lambda here has two purposes: 1) makes the os.path.join a function as first arg of map()
                    # 2) provides an argument x for os.path.join(root_folder, sub_folder, x=image)
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
                    # here we're creating a list of all the path to the images
                    img_list.extend(images_path)
                    # same thing for the labels
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    labels.extend(lab_path)

                    # what if we wanted to create the labels for a simple classification task?
                    #lab = [seq.strip() for i in range(len(os.listdir(os.path.join
                    #      (db_root_dir, "Annotations/Full-Resolution", seq.strip()))))]
                    #labels.extend(lab)
        else:

            # retrieves just one img and mask of a specified class (seq_name)
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))

            img_list = list(map(lambda x: os.path.join('JPEGImages/Full-Resolution/', str(seq_name), x), names_img))
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
            labels = [os.path.join('Annotations/480p/', str(seq_name), name_label[0])]

            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]
                
        print(len(labels), len(img_list))
        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # idx order is controlled by torch.utils.data.DataLoader(shuffle): if shuffle = True, idx will be
        # retrieved randomly, otherwise they will be sequential from 0 to __len__
        img = np.array(Image.open(os.path.join(self.db_root_dir, self.img_list[idx])).convert("RGB"), dtype=np.float32)
        gt = np.array(Image.open(os.path.join(self.db_root_dir, self.labels[idx])).convert("L"), dtype=np.float32)
        
        gt = ((gt/np.max([gt.max(), 1e-8])) > 0.5).astype(np.float32)
        
        if self.transform is not None:

            augmentations = self.transform(image=img, mask=gt)
            img = augmentations["image"]
            gt = augmentations["mask"]
            
        # if image width and height is < than expected shape --> we should apply mirroring:
        # with padding_mode="reflect"
        # https://pytorch.org/vision/0.12/generated/torchvision.transforms.Pad.html
        if self.pad_mirroring:
            img = Pad(padding=self.pad_mirroring, padding_mode="reflect")(img)

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))
        return list(img.shape[:2])


if __name__ == '__main__':
    import torch
    from matplotlib import pyplot as plt
    from utils import inv_normalize, tens2image


    # sys.getsizeof(getattr(dataset, "img_list"))
    # to check the size the list containing all the path to images

    dataset = DAVIS2017(db_root_dir=ROOT_DIR, train=True,
                        transform=train_transform, pad_mirroring=PAD_MIRRORING)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    assert getattr(dataloader, "batch_size") == 1, "The plot script works only with batch_size=1"

    for i, (img, gt) in enumerate(dataloader):
        img = CenterCrop((IMAGE_HEIGHT, IMAGE_WIDTH))(img)
        plt.figure()
        plt.imshow(overlay_mask(inv_normalize(tens2image(img)), tens2image(gt)))

        if i == 20:
            break

    plt.show(block=True)
