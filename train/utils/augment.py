import albumentations as A
import cv2
import os
from tqdm import tqdm

def get_filters():
    filters = [
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
        A.Emboss(p=0.2)]
    return filters


class Augmentor:

    _file_counter = 0

    def __init__(self, data_dir, output_dir, augment_params):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.augment_params = augment_params
        self.filters = get_filters()


    @staticmethod
    def save_augmented_image(image, path):
        cv2.imwrite(path, image)

    def new_augment(self, image_path):
        image = cv2.imread(image_path)
        image = self.aug(image=image)['image']
        return image

    def auto_augment(self, quantity, resize=False, width=0, height=0):
        if resize:
            self.filters.insert(0, A.Resize(width=width, height=height, p=1))

        self.aug = A.Compose(self.filters)

        images_list = os.listdir(self.data_dir)
        images_list.sort()

        for image_name in tqdm(images_list):

            image_path = os.path.join(self.data_dir, image_name)

            for _ in range(quantity):
                image = self.new_augment(image_path)
                new_image_path = os.path.join(self.output_dir, f'{self._file_counter}.png')
                self.save_augmented_image(image, new_image_path)
                self._file_counter += 1
