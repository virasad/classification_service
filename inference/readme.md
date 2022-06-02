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

## Send Request
### Send request to predict image
```json
{
    "image": "base62 encoded image"
}
```

## Set Model
### Send request to set model
```json
{
  "model_path": "./path/to/model.pth"
}
```