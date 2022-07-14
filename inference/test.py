import os

import requests


def set_model(model_path):
    url = 'http://127.0.0.1:3000/set-model'
    model_path = os.path.abspath(model_path)
    print(model_path)
    res = requests.post(url, params={"model_path": model_path}).json()
    print(res)


def inference_request(image_p):
    url = 'http://127.0.0.1:3000/predict'
    image_p = os.path.abspath(image_p)
    image_data = open(image_p, "rb").read()
    res = requests.post(url, files={"image": image_data}).json()
    print(res)


def main():
    image_p = '../dataset/predict/72100438_73de9f17af.jpg'
    model_path = '../train/weights/hymenoptera_data_model.pt'
    set_model(model_path)
    inference_request(image_p)


main()
