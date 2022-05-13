import os
import shutil
import random


def split_to_train_val(dataset_path, train_val_ratio):
    """
    Split dataset to train and val.
    if train folder doesn't exist, create it and if exists, delete it and create it again.
    if val folder doesn't exist, create it and if exists, delete it and create it again.
    crate folder for each class in train and val.
    """

    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    if not os.path.exists(train_path):
        os.mkdir(train_path)
    else:
        shutil.rmtree(train_path)
        os.mkdir(train_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    else:
        shutil.rmtree(val_path)
        os.mkdir(val_path)

    for classes in os.listdir(dataset_path):
        if not os.path.exists(os.path.join(train_path, classes)):
            os.mkdir(os.path.join(train_path, classes))
        else:
            shutil.rmtree(os.path.join(train_path, classes))
            os.mkdir(os.path.join(train_path, classes))
        if not os.path.exists(os.path.join(val_path, classes)):
            os.mkdir(os.path.join(val_path, classes))
        else:
            shutil.rmtree(os.path.join(val_path, classes))
            os.mkdir(os.path.join(val_path, classes))

    for classes in os.listdir(dataset_path):
        for video in os.listdir(os.path.join(dataset_path, classes)):
            if random.random() < train_val_ratio:
                shutil.copy(os.path.join(dataset_path, classes, video), os.path.join(train_path, classes))
            else:
                shutil.copy(os.path.join(dataset_path, classes, video), os.path.join(val_path, classes))

    return train_path, val_path

def copy(a, b):
    """Copy all contents of directory a to directory b."""
    for f in os.listdir(a):
        if os.path.isdir(os.path.join(a, f)):
            shutil.copytree(os.path.join(a, f), os.path.join(b, f))
        else:
            shutil.copy2(os.path.join(a, f), os.path.join(b, f))
