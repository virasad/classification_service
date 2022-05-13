from torchmetrics import F1, Accuracy, Precision, Recallimport

import torch
import os
import shutil
import flash
from flash.image import ImageClassifier
from utils import dataloader, utils, logger, augment


class ClassificationTrainer:
    def __init__(self, backbone, pre_trained_path=None, is_augment=False, augment_params=None, epochs=100, batch_size=4,
                 num_dataloader_workers=8, num_classes=2):
        self.backbone = backbone
        self.is_augment = is_augment
        self.augment_params = augment_params
        self.pre_trained_path = pre_trained_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.num_classes = num_classes

        pass

    def train_from_images(self, train_path, val_path, save_name):

        datamodule = dataloader.get_dataset_for_flash(train_path, val_path, self.batch_size,
                                                      self.num_dataloader_workers)

        if self.pre_trained_path != None:
            model = ImageClassifier.load_from_checkpoint(self.pre_trained_path)
        else:
            model = ImageClassifier(backbone=self.backbone,
                                    labels=datamodule.labels,
                                    num_classes=self.num_classes,
                                    metrics=[
                                        F1(num_classes=datamodule.num_classes,
                                           mdmc_average='samplewise'),
                                        Accuracy(num_classes=datamodule.num_classes,
                                                 mdmc_average='samplewise'),
                                        Precision(num_classes=datamodule.num_classes,
                                                  mdmc_average='samplewise'),
                                        Recall(num_classes=datamodule.num_classes,
                                               mdmc_average='samplewise')]
                                    )

        trainer = flash.Trainer(
            max_epochs=self.epochs, logger=logger.ClientLogger(), gpus=torch.cuda.device_count())
        trainer.finetune(model, datamodule=datamodule, strategy="no_freeze"),
        trainer.save_checkpoint(
            os.path.join(os.environ.get('WEIGHTS_DIR', '/weights'), "{}_model.pt".format(save_name)))
        result = trainer.validate(model, datamodule=datamodule)

        return result[0]

    def pre_process_train(self, images_path, validation_split=0.2, save_name=''):
        train_path, val_path = utils.split_to_train_val(images_path, validation_split)

        # create temp folder and if exist delete it and create a new one
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        os.mkdir('temp')

        # create train and val folder in temp
        os.mkdir('temp/train')
        os.mkdir('temp/val')

        # copy train and val images to train and val folder
        utils.copy(train_path, 'temp/train')
        utils.copy(val_path, 'temp/val')

        train_path = 'temp/train'
        val_path = 'temp/val'

        if self.is_augment:
            train_path_augmented = 'temp/train_augmented'
            val_path_augmented = 'temp/val_augmented'

            # Augment train
            aug = augment.Augmentor(train_path, train_path_augmented, self.augment_params)
            for _class in os.listdir(train_path):
                each_class_path = os.path.join(train_path_augmented, _class)
                os.mkdir(each_class_path)
                aug.auto_augment(self.augment_params)

            # Augment val
            aug = augment.Augmentor(val_path, val_path_augmented, self.augment_params)
            for _class in os.listdir(val_path):
                each_class_path = os.path.join(val_path_augmented, _class)
                os.mkdir(each_class_path)
                aug.auto_augment(self.augment_params)

            train_path = train_path_augmented
            val_path = val_path_augmented

        # train and val images are ready
        result = self.train_from_images(train_path, val_path, save_name)
        return result
