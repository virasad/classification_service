import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
import trainer as tr

app = FastAPI()


class ClassificationTrain(BaseModel):
    images : str
    save_name : Optional[str] = None
    batch_size : Optional[str] = None
    extra_kwargs : Optional[dict] = None
    num_dataloader_workers : Optional[str] = None
    epochs : Optional[str] = None
    num_classes : str = 2
    validation_split : str = 0.2
    pretrained_path : Optional[str] = None
    is_augment : Optional[bool] = None
    augment_params : Optional[dict] = None
    backbone : Optional[str] = None

@app.post("/train/")
def train(classification : ClassificationTrain):
    try:
        trainer = tr.ClassificationTrainer(
            backbone=classification.backbone,
            pre_trained_path=classification.pretrained_path,
            is_augment=classification.is_augment,
            augment_params=classification.augment_params,
            epochs=classification.epochs,
            batch_size=classification.batch_size,
            num_dataloader_workers=classification.num_dataloader_workers,
            num_classes=classification.num_classes)

        result = trainer.pre_process_train(
            images_path=classification.images,
            validation_split=classification.validation_split,
            save_name=classification.save_name)

        response_url = os.environ.get('RESPONSE_URL', 'http://127.0.0.1:8000/api/v1/train/done')
        a = requests.post(response_url,
                          data={**result, **train.extra_kwargs, 'save_name': train.save_name + '_model.pt'})
        return result

    except Exception as e:
        return {"result": "failed", 'error': str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get('PORT', '5554')))