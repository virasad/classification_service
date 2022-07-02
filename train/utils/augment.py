import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from flash import InputTransform
from flash.image.classification.input_transform import AlbumentationsAdapter
from typing import Callable, Tuple, Union
from dataclasses import dataclass


@dataclass
class TrainTransform(InputTransform):
    def __init__(self, image_w=64, image_h=64, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.image_w = image_w
        self.image_h = image_h
        self.mean = mean
        self.std = std

    def input_per_sample_transform(self):
        return A.Compose([
            A.Resize(height=self.image_h, width=self.image_w),
            A.Normalize(p=1, std=self.std, mean=self.mean),
            ToTensorV2(p=1)
        ])

    def train_input_per_sample_transform(self):
        return A.Compose([
            A.Resize(height=self.image_h, width=self.image_w),
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
