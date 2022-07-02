from flash.image import ImageClassificationData
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from flash import InputTransform
from typing import Callable, Tuple, Union
from dataclasses import dataclass
import numpy as np


class AlbumentationsAdapter(torch.nn.Module):
    def __init__(self, transform):
        super().__init__()
        if not isinstance(transform, list):
            transform = [transform]
        self.transform = A.Compose(transform)

    def forward(self, x):
        res = self.transform(image=np.asarray(x))["image"]
        return res


@dataclass
class TrainTransform(InputTransform):
    image_size: Tuple[int, int] = (196, 196)
    mean: Union[float, Tuple[float, float, float]] = (0.485, 0.456, 0.406)
    std: Union[float, Tuple[float, float, float]] = (0.229, 0.224, 0.225)

    def input_per_sample_transform(self):
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(p=1, std=self.std, mean=self.mean),
            ToTensorV2(p=1)
        ])

    def train_input_per_sample_transform(self):
        return AlbumentationsAdapter([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0,
                               shift_limit=0.1, p=1, border_mode=0),
            A.RandomRotate90(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomShadow(p=0.1),
            A.RandomSnow(snow_point_lower=0.1,
                         snow_point_upper=0.15, p=0.1),
            A.RGBShift(p=0.2),
            A.CLAHE(p=0.2),

            A.HueSaturationValue(
                p=0.1, hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),

            A.MotionBlur(p=0.1),
            A.MedianBlur(p=0.2),
            A.ISONoise(p=0.2),
            A.Posterize(p=0.2),
            A.Perspective(p=0.1),
            A.PiecewiseAffine(
                p=0.1, scale=(0.01, 0.02)),
            A.Emboss(p=0.2),
            A.Normalize(p=1, std=self.std, mean=self.mean),
            ToTensorV2(p=1)
        ])

    def target_per_sample_transform(self) -> Callable:
        return torch.as_tensor


def get_dataset_for_flash(dataset_path, batch_size=2, num_workers=8, image_width=128, image_height=128, split=0.2):
    datamodule = ImageClassificationData.from_folders(
        train_folder=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_kwargs={"image_size": (image_width, image_height), "mean": (0.485, 0.456, 0.406),
                          "std": (0.229, 0.224, 0.225)},
        train_transform=TrainTransform,
        val_split=split
    )
    print(datamodule.num_classes)

    return datamodule


if __name__ == '__main__':
    get_dataset_for_flash('/home/amir/Projects/classification_service/dataset/all')
