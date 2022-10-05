import tensorflow as tf
import os

_URL= "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip= tf.keras.utils.get_file(
    fname= 'cats_and_dogs.zip',
    origin= _URL,
    extract= True
)
PATH= os.path.join(
    os.path.dirname(path_to_zip), 'cats_and_dogs_filtered'
)

train_dir= os.path.join(
    PATH, 'train'
)

val_dir= os.path.join(
    PATH, 
    'validation'
)

BATCH_SIZE= 32
IMG_SIZE= (160, 160)



def main():

    train_dataset= tf.keras.utils.image_dataset_from_directory(
        directory= train_dir,
        shuffle= True,
        batch_size= BATCH_SIZE,
        image_size= IMG_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory= val_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    return train_dataset, validation_dataset

if __name__ == "__main__":
    train, val= main()