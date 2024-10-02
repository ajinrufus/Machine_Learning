'''
Functionality for creating pyTorch datasets and dataloader for image classification'''
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# number of cores - useful for performing parallel jobs
n_workers = os.cpu_count()

def create_dataset_dataloader(train_dir : str, test_dir : str,
                              transform : transforms.Compose,
                              batch_size : int, n_workers : int = n_workers):
    '''Creates training and testing dataloaders

    Args:
        train_dir : path of training data
        test_dir  : path of testing data
        transform : transformations to be performed
        batch_size: mini-batch size
        n_workers : number of cores per dataloader

    Return:
        A tuple of train_dataloader, test_dataloader, class_names
        train_dataloader : dataloader of training data
        test_dataloader  : dataloader of testing data
        class_names      : list of label class names'''


    # imageFolder to create datasets
    # training dataset
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)

    # testing dataset
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # class names
    class_names = train_data.classes

    # dataset to dataloader
    # training dataloader
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=n_workers, 
                                  pin_memory=True) # pin_memory - quicker transfer from cpu to gpu

    # testing dataloader
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                                  shuffle=False, num_workers=n_workers, 
                                  pin_memory=True) # no need to shuffle

    return train_dataloader, test_dataloader, class_names
