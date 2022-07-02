import os

import click
import flash
import torch
from flash.image import ImageClassifier
from torchmetrics import Accuracy, Precision, Recall
from torchmetrics.classification.f_beta import F1Score

from utils import dataloader, logger


class ClassificationTrainer:
    def __init__(self, backbone='mobilenet_v2', pre_trained_path=None, epochs=100, batch_size=2,
                 num_dataloader_workers=8, image_width=256, image_height=256,
                 validation_split=0.2):
        self.backbone = backbone
        self.pre_trained_path = pre_trained_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.image_width = image_width
        self.image_height = image_height
        self.validation_split = validation_split

    def trainer(self, dataset_path, save_name=''):
        datamodule = dataloader.get_dataset_for_flash(dataset_path, self.batch_size,
                                                      self.num_dataloader_workers,
                                                      image_width=self.image_width,
                                                      image_height=self.image_height,
                                                      split=self.validation_split)
        if self.pre_trained_path:
            model = ImageClassifier.load_from_checkpoint(self.pre_trained_path)
        else:
            model = ImageClassifier(backbone=self.backbone,
                                    labels=datamodule.labels,
                                    num_classes=datamodule.num_classes,
                                    metrics=[
                                        F1Score(num_classes=datamodule.num_classes),
                                        Accuracy(num_classes=datamodule.num_classes),
                                        Precision(num_classes=datamodule.num_classes),
                                        Recall(num_classes=datamodule.num_classes)]
                                    )

        trainer = flash.Trainer(
            max_epochs=self.epochs, logger=logger.ClientLogger(), gpus=torch.cuda.device_count())
        trainer = flash.Trainer(
            max_epochs=self.epochs, gpus=torch.cuda.device_count())
        trainer.finetune(model, datamodule=datamodule, strategy="no_freeze"),
        save_path = os.path.join(os.environ.get('WEIGHTS_DIR', './weights'), "{}_model.pt".format(save_name))
        trainer.save_checkpoint(save_path)
        result = trainer.validate(model, datamodule=datamodule)
        result[0]['model_p'] = save_path
        return result[0]['model_p']


@click.command()
@click.option('--dataset_path', default='./dataset/', help='Path to dataset')
@click.option('--save_name', default='', help='Name of the model')
@click.option('--epochs', default=100, help='Number of epochs')
@click.option('--batch_size', default=1, help='Batch size')
@click.option('--num_dataloader_workers', default=8, help='Number of dataloader workers')
@click.option('--image_width', default=128, help='Image width')
@click.option('--image_height', default=128, help='Image height')
@click.option('--validation_split', default=0.2, help='Validation split')
@click.option('--backbone', default='mobilenet_v2', help='Backbone')
@click.option('--pre_trained_path', default=None, help='Path to pre-trained model')
def cli(dataset_path, save_name, epochs, batch_size, num_dataloader_workers, image_width, image_height,
        validation_split, backbone, pre_trained_path):
    trainer = ClassificationTrainer(backbone=backbone,
                                    pre_trained_path=pre_trained_path,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    num_dataloader_workers=num_dataloader_workers,
                                    image_width=image_width,
                                    image_height=image_height,
                                    validation_split=validation_split)
    result = trainer.trainer(dataset_path, save_name)
    print(result)
    return result


if __name__ == '__main__':
    cli()
