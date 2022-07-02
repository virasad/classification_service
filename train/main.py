import os
from enum import Enum
from typing import Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel

import trainer as tr
import torch

app = FastAPI()


# TODO add more models
class ModelName(str, Enum):
    mobilenet_v2 = "mobilenet_v2"
    todo1 = 'todo1'
    todo2 = 'too2'


class ClassificationTrain(BaseModel):
    dataset_p: str
    save_name: Optional[str] = ''
    batch_size: Optional[str] = 4
    extra_kwargs: Optional[dict] = None
    num_dataloader_workers: Optional[str] = 8
    epochs: Optional[str] = 100
    validation_split: str = 0.2
    pretrained_path: Optional[str] = None
    backbone: Optional[ModelName] = ModelName.mobilenet_v2.value
    image_width: Optional[int] = 256
    image_height: Optional[int] = 256


@app.post("/train/")
def train(dataset_p: str,
          save_name: Optional[str] = '',
          batch_size: Optional[int] = 4,
          num_dataloader_workers: Optional[int] = 8,
          epochs: Optional[int] = 100,
          validation_split: float = 0.2,
          pretrained_path: Optional[str] = None,
          backbone: Optional[ModelName] = ModelName.mobilenet_v2.value,
          image_width: Optional[int] = 256,
          image_height: Optional[int] = 256,
          response_url: Optional[str] = None,
          extra_kwargs: Optional[dict] = None,
          ):
    try:
        trainer = tr.ClassificationTrainer(
            backbone=backbone,
            pre_trained_path=pretrained_path,
            epochs=epochs,
            batch_size=batch_size,
            num_dataloader_workers=num_dataloader_workers,
            image_width=image_width,
            image_height=image_height,
            validation_split=validation_split
        )

        result = trainer.trainer(
            dataset_path=dataset_p,
            save_name=save_name)
        # release GPU memory
        del trainer
        torch.cuda.empty_cache()
        if response_url:
            requests.post(response_url,
                              data={**result, **extra_kwargs, 'save_name': train.save_name + '_model.pt'})
        return result

    except Exception as e:
        return {"result": "failed", 'error': str(e)}
