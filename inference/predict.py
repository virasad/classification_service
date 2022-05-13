import warnings

from flash import Trainer
import torch
from PIL import Image
import numpy as np
from flash.image import ImageClassificationData, ImageClassifier

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class Predictor:
    def __init__(self, classes):
        self.trainer = Trainer(gpus=torch.cuda.device_count())
        self.num_classes = len(classes)
        self.classes = classes

    def load_model(self, model_path):
        self.model = ImageClassifier.load_from_checkpoint(model_path)


    def predict(self, images, batch_size):

        if isinstance(images, list):
            images = [images]

        if isinstance(images[0], str):
            images = [np.array(Image.open(image)) for image in images]
            images = [np.moveaxis(image, 2, 0) for image in images]

        datamodule = ImageClassificationData.from_numpy(predict_data=images,
                                                         batch_size=batch_size,
                                                         num_classes=self.num_classes)
        predictions = self.trainer.predict(self.model, datamodule=datamodule, output='probabilities')

        classes = []
        for prediction in predictions[0]:
            classes.append(self.classes[np.argmax(prediction)])

        probabilities = []
        for prediction in predictions[0]:
            probabilities.append(max(prediction))

        result = {
            'classes': classes,
            'probabilities': probabilities
        }
        return result


