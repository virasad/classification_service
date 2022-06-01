# Installation
First of all please install the dependencies:
- [pytorch](https://pytorch.org/get-started/locally/)
- [Flash images](https://lightning-flash.readthedocs.io/en/latest/installation.html)
- Flash:

`pip install lightning-flash`
## Addition for webservice

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

```bash
pip install 'lightning-flash[image]'
```
- other dependencies:
```bash
pip install -r requirements.txt 
```

# How to run?
## run from script

```bash
uvicorn main:app --reload
```


## response url
set `RESPONSE_URL` in environment variables to your response url

## Send Train request

```json
{
  "image": "Images Path",
  "save_name": "Save Weights Name",
  "epochs": "Epochs Number",
  "batch_size": "Batch Size Number",
  "num_dataloader_workers": "Number of workers for dataloader",
  "num_classes": "Number of classes",
  "validation_split": "Validation Split Number to split train and validation",
  "pretrained_path": "Pretrained Weight Path", -required if you want to train from scratch
  "is_augment": "True or False for augmentation",
  "augment_params" : "Augmentation Params", -required if is_augmentation is True
  "backbone": "Backbone Name"
}
```