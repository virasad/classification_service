from flash.image import ImageClassificationData

def get_dataset_for_flash(train_path, val_path, batch_size, num_workers):
    datamodule = ImageClassificationData.from_folders(
        train_folder= train_path,
        val_folder= val_path,
        batch_size= batch_size,
        num_workers=num_workers
    )
    return datamodule