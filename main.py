import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from os.path import exists
from os import rename, stat
from shutil import rmtree, make_archive, move
from statistics import mean
from imghdr import what
from random import uniform
import numpy as np
import pathlib
import PIL
import contextlib
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
logging.getLogger("tensorflow").setLevel(logging.ERROR)

print(f"tensorflow version is {tf.__version__}, devices detected: {tf.config.list_physical_devices()}\n")

train = True if input("would you like to train the model or predict an image? (1 or 2): ").lower() == "1" else False

print("\n", end="")
old_model = "models/old_model"
latest_model = "models/latest_model"

batch_size = 32
img_height = 180
img_width = 180


# noinspection PyShadowingNames
def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is a symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def before_exit():
    save = True if input("would you like to save your trained model? (y or n): ") \
                       .lower() == "y" else False
    if not save:
        exit()

    if exists(old_model) and exists(latest_model):
        rmtree(old_model)
        rename(latest_model, old_model)
    elif exists(latest_model):
        rename(latest_model, old_model)

    tf.keras.models.save_model(model, latest_model)
    make_archive(latest_model, 'zip', latest_model)

    print(f"\nok, saved to {latest_model} ({round(get_size(latest_model) / 1000000, 1)} MBs).")


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if not exists("dataset"):
    print("dataset not found. (folder name should be 'dataset' and in this directory)")
    exit()

data_dir = pathlib.Path("dataset")

clean = True if input("would you like to clean your dataset? (y or n): ").lower() == "y" else False
print("\n", end="")

if clean:
    files = 0
    total = sum([len(files) for r, d, files in os.walk("dataset")])

    for dirpath, dirnames, filenames in os.walk("dataset"):
        for f in filenames:
            files += 1

            fp = os.path.join(dirpath, f)
            path = pathlib.Path(fp)
            new_path = os.path.join("invalid_files", path.name if "dataset" in path.parent.name else
                                    os.path.join(str(path.parent).replace("dataset\\", ""), path.name))

            if what(fp) is None:
                pathlib.Path(new_path.replace(pathlib.Path(new_path).name, "")).mkdir(exist_ok=True, parents=True)
                move(fp, new_path)

            print(f"\rcleaning dataset.. ({files}/{total})", end="")
    print("\rcleaning dataset..done")

print("loading dataset..", end="")

with suppress_stdout():
    # noinspection PyUnboundLocalVariable
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    print(f"\rloading dataset.. (1/2)", end="")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    print("\rloading dataset..done")

class_names = train_ds.class_names
print("done\n")

print(f"found {len(class_names)} labels from dataset.\n")

if train:
    normalization_layer = layers.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    num_classes = len(class_names)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(uniform(0.1, 0.4)),
            layers.RandomZoom(uniform(0.1, 0.4)),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

    if exists(latest_model):
        load = True if input(
            "would you like to load your latest saved model to continue training? (y or n): ").lower() == "y" else False
        if load:
            print("loading model..", end="")
            model = tf.keras.models.load_model(latest_model, compile=False)
            print("done")

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    try:
        EPOCHS = int(input("how many epochs do you want to train for? "))
        if EPOCHS < 0:
            print("cannot train for 0 epochs, falling back to 5 epochs.\n")
            EPOCHS = 5
    except ValueError:
        print("not a number, falling back to 5 epochs.\n")
        EPOCHS = 5

    for i in range(EPOCHS):
        print("", end="\n\n" if i == 0 else "\n")

        print("Epoch 1: ")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1
        )

    before_exit()
else:
    model = None
    if exists(latest_model):
        print("loading model..", end="")
        model = tf.keras.models.load_model(latest_model, compile=False)
        print("done")
    else:
        print("no model found. please train the model first")
        exit()

    running = True

    while running:
        try:
            img = tf.keras.utils.load_img(
                input("\ninput image path: "), target_size=(img_height, img_width)
            )

            print("\n", end="")
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            print(
                "This image most likely belongs to {} with a {:.2f}% confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
            )
        except (FileNotFoundError, OSError):
            print("file not found\n")
            pass
        except KeyboardInterrupt:
            exit()
