import warnings

import click
import numpy as np
import torch
from PIL import Image
from flash import Trainer
from flash.image import ImageClassificationData, ImageClassifier

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class Predictor:
    def __init__(self):
        self.trainer = Trainer(gpus=torch.cuda.device_count())

    def load_model(self, model_path):
        self.model = ImageClassifier.load_from_checkpoint(model_path)

    def predict(self, images, batch_size):

        if not isinstance(images, list):
            images = [images]

        if isinstance(images[0], str):
            print('Loading images from disk')
            print(images)
            images = [np.asarray(Image.open(image)) for image in images]
            images = [np.moveaxis(image, 2, 0) for image in images]

        if isinstance(images[0], np.ndarray):
            images = [np.moveaxis(image, 2, 0) for image in images]

        datamodule = ImageClassificationData.from_numpy(predict_data=images,
                                                        batch_size=batch_size,
                                                        )
        predictions = self.trainer.predict(self.model, datamodule=datamodule, output='labels')
        return predictions[0]


@click.command()
@click.option('--model_path', default='./weights/model.pt', help='Path to model')
@click.option('--images', default='./images/test.jpg', help='Path to image')
@click.option('--batch_size', default=1, help='Batch size')
def cli(model_path, images, batch_size):
    import pprint
    predictor = Predictor()
    predictor.load_model(model_path)
    result = predictor.predict(images, batch_size)
    pprint.pprint(result)


if __name__ == '__main__':
    cli()
