import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler

# this function takes 3 arguments: the directory of your image folder, the % of data used for validation,
# the library that can be either tensorflow or pytorch

def load_imagefolder(dir, val_size=0.1, library = "tf"):
    assert library in ["tf", "pt"], "choose between 'tf' and 'pt'"
    if library == "tf":

        # we create our BatchDataset objects that are iterable objects composed by tuples
        # whose dimensions are: a, b = next(iter(train_loader)) --> a = (256,3,28,28), b=(256,)
        # (a-images, b-labels) --Z notice that the tensor related to images have shape
        # (batch_size, height, width, channel)
        train_loader = image_dataset_from_directory(dir, batch_size=64, image_size=(32, 32),
                                                     validation_split=val_size, subset="training",
                                                     seed=123)
        valid_loader = image_dataset_from_directory(dir, batch_size=64, image_size=(32, 32),
                                                   validation_split=val_size, subset="validation",
                                                   seed=123)

        # on tensorflow the method is very simple, but you have less control overall
    else:

        # on Pytorch the procedure is less automated:

        # transform --> here we create a transform.Compose object to be pass as argument to the image folder
        # the only one that is mandatory is the transforms.ToTensor() since by default the datasets.ImageFolder
        # loads images as PIL objects which would later on trigger errors.
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        ])
                                        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_images = datasets.ImageFolder(dir, transform=transform)

        # following lines are quite common and shuffle the images based on their indices
        # and split them for train and validation

        valid_size = val_size
        num_train = len(train_images)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        # we create our Dataloades obj that are tuples iterable objects whose dimensions are:
        # a, b = next(iter(train_loader)) --> a = (256,3,28,28), b=(256,)
        # notice the difference between tensorflow and pytorch: tf: a = (256,28,28,3) - py:  a = (256,3,28,28)
        train_loader = DataLoader(train_images, batch_size=64, sampler=train_sampler,
                                  num_workers=0)
        valid_loader = DataLoader(train_images, batch_size=64, sampler=valid_sampler,
                                  num_workers=0)

    return train_loader, valid_loader


