import base64
import json

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from predict import Predictor
import os

classes = ["", ""]

app = FastAPI()
detector = Predictor(classes)

class Predict(BaseModel):
    image: str

class ModelPath(BaseModel):
    model_path: str


@app.post('/set_model/')
def set_model(model_path: ModelPath):
    try:
        detector.load_model(model_path.model_path)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post('/predict/')
def predict(predict: Predict):
    try:
        image = cv2.imdecode(np.frombuffer(base64.b64decode(predict.image.encode('utf-8')), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = np.moveaxis(image, 2, 0)
        result = detector.predict(image, batch_size=1)
        result = json.dump(result)
        return result

    except Exception as e:
        return {"result": "failed", 'error': str(e)}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('PORT', '5556')))
