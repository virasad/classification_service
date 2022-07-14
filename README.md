# Classification Service

Train Classification model as a service. **Easy as ABC!**

The goal of this project is to train and test a classification model without any code or knowledge of the deeplearning.

# Usage

## Docker Compose

### Install dependencies

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Docker gpu

```bash
# It should be install for gpu support
bash docker-gpu-installation.sh
```

```bash
docker-compose up -d
```

---

## Shell

### Install dependencies

- [Pytorch & TorchVision](https://pytorch.org/get-started/locally/)

```bash
cd train
pip install -r requirements.txt

cd inference
pip install -r requirements.txt
```

---

# Train

## Prepare data

Put your dataset in the following format:

```bash
- dataset
  - label
    - image.jpg
    - image.jpg
    - image.jpg
      ...
  - label2
    - image.jpg
    - image.jpg
    - image.jpg
      ...
  - labeln
    - image.jpg
    - image.jpg
    - image.jpg
      ...
      
```

## Train from bash

```bash
cd train
python trainer.py --dataset_path {YOUR_DATASET_PATH} \
                  --save_name {YOUR_DATASET_PATH} \
                  --epochs {YOUR_EPOCHS} \
                  --batch_size {YOUR_BATCH_SIZE} \
                  --num_dataloader_workers {YOUR_NUM_DATALOADER_WORKERS} \
                  --image_width {NETWORK_INPUT_WIDTH} \
                  --image_height {NETWORK_INPUT_HEIGHT} \
                  --validation_split {YOUR_VALIDATION_SPLIT} \
                  --pretrained_path {YOUR_PRETRAINED_PATH} \
                  --backbone {YOUR_BACKBONE} 
```

## Train with API

If you run the docker compose file you should put your dataset in volumes/dataset folder and your weights in
volumes/weights folder.

### Parameters

```json
{
  "dataset_p": dataset_p,
  "save_name": save_name,
  "batch_size": batch_size,
  "num_dataloader_workers": num_dataloader_workers,
  "epochs": epochs,
  "validation_split": validation_split,
  "pretrained_path": pretrained_path,
  "backbone": backbone,
  "image_width": image_width,
  "image_height": image_height,
  "response_url": response_url,
  "extra_kwargs": extra_kwargs
}
```

- **response_url** is the url to send the response. It can be Null or not send

- **extra_kwargs** is a dictionary that will send back to your response API after the train is finished.

### Example

- For train example refer to the [example](train/test.py)
- For inference example refer to the [example](inference/test.py)
