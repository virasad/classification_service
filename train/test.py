import os
import tarfile
import zipfile
from pathlib import Path

import requests
import urllib3
from tqdm.auto import tqdm as tq


def download_data(url: str, path: str = "dataset/", verbose: bool = False) -> None:
    """Download file with progressbar.
    # __license__ = "MIT"
    """
    # Disable warning about making an insecure request
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if not os.path.exists(path):
        os.makedirs(path)
    local_filename = os.path.join(path, url.split("/")[-1])
    r = requests.get(url, stream=True, verify=False)
    file_size = int(r.headers["Content-Length"]) if "Content-Length" in r.headers else 0
    chunk_size = 1024
    num_bars = int(file_size / chunk_size)
    if verbose:
        print(dict(file_size=file_size))
        print(dict(num_bars=num_bars))

    if not os.path.exists(local_filename):
        with open(local_filename, "wb") as fp:
            for chunk in tq(
                    r.iter_content(chunk_size=chunk_size),
                    total=num_bars,
                    unit="KB",
                    desc=local_filename,
                    leave=True,  # progressbar stays
            ):
                fp.write(chunk)

    def extract_tarfile(file_path: str, extract_path: str, mode: str):
        if os.path.exists(file_path):
            with tarfile.open(file_path, mode=mode) as tar_ref:
                for member in tar_ref.getmembers():
                    try:
                        tar_ref.extract(member, path=extract_path, set_attrs=False)
                    except PermissionError:
                        raise PermissionError(f"Could not extract tar file {file_path}")

    if ".zip" in local_filename:
        if os.path.exists(local_filename):
            with zipfile.ZipFile(local_filename, "r") as zip_ref:
                zip_ref.extractall(path)
    elif local_filename.endswith(".tar.gz") or local_filename.endswith(".tgz"):
        extract_tarfile(local_filename, path, "r:gz")
    elif local_filename.endswith(".tar.bz2") or local_filename.endswith(".tbz"):
        extract_tarfile(local_filename, path, "r:bz2")


def main():
    url = 'http://127.0.0.1:3000/train'
    save_dir = Path("./data")
    dataset_p = save_dir / 'hymenoptera_data'
    all_dataset_p = dataset_p / 'all'
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip",
                  "./data"
                  , verbose=True
                  )
    Path.mkdir(all_dataset_p, exist_ok=True)
    # move from train and test and val folder to all folder
    for folder in ["train", "test", "val"]:
        folder_p = dataset_p / folder
        for file in folder_p.iterdir():
            try:
                file.rename(all_dataset_p / file.name)
            except Exception as e:
                print(e)
                continue
    request_train(
        url=url,
        dataset_p=str(all_dataset_p),
        save_name="hymenoptera_data",
        batch_size=4,
        num_dataloader_workers=8,
        epochs=100,
        validation_split=0.2,
        pretrained_path=None,
        backbone="mobilenet_v2",
        image_width=256,
        image_height=256,
        response_url=None)


def request_train(url,
                  dataset_p="dataset/hymenoptera_data/all",
                  save_name='sample',
                  batch_size=2,
                  num_dataloader_workers=8,
                  epochs=100,
                  validation_split=0.2,
                  pretrained_path=None,
                  backbone="mobilenet_v2",
                  image_width=256,
                  image_height=256,
                  response_url=None,
                  extra_kwargs=None):
    # dataset_p to complete path to absolute path
    dataset_p = os.path.abspath(dataset_p)
    print(dataset_p)
    payload = {
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
    res = requests.post(url, params=payload)
    print(res.text)


if __name__ == "__main__":
    main()
