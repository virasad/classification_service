import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile

from predict import Predictor

app = FastAPI()
detector = Predictor()


@app.post('/set-model/')
def set_model(model_path: str):
    try:
        detector.load_model(model_path)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post('/predict/')
async def predict(image: UploadFile = File(...), ):
    try:
        contents = await image.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = detector.predict(img, batch_size=1)
        return result

    except Exception as e:
        return {"result": "failed", 'error': str(e)}
