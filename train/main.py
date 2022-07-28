from enum import Enum
from typing import Optional

from fastapi import FastAPI, BackgroundTasks

import trainer as tr

app = FastAPI()


# TODO add more models
class ModelName(str, Enum):
    mobilenet_v2 = "mobilenet_v2"
    todo1 = 'todo1'
    todo2 = 'too2'


@app.post("/train/")
def train(dataset_p: str,
          background_tasks: BackgroundTasks,
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
          log_url: Optional[str] = None,
          extra_kwargs: Optional[dict] = None,
          ):
    try:
        background_tasks.add_task(tr.ClassificationTrainer(
            backbone=backbone.value,
            pre_trained_path=pretrained_path,
            epochs=epochs,
            batch_size=batch_size,
            num_dataloader_workers=num_dataloader_workers,
            image_width=image_width,
            image_height=image_height,
            validation_split=validation_split,
            response_url=response_url,
            log_url=log_url,
            extra_kwargs=extra_kwargs
        ).trainer, dataset_path=dataset_p, save_name=save_name)
        return {"status": "ok"}

    except Exception as e:
        return {"result": "failed", 'error': str(e)}
