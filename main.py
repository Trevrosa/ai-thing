from packaging import version
local_ver = "0.2.4"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import cv2 as cv
import contextlib
import logging
import json
import requests

from pathlib import Path
from PIL import Image
from os import remove
from os.path import exists
from shutil import rmtree, make_archive, move
from imghdr import what
from random import uniform

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").disabled = True

local_ver = version.parse(local_ver)
remote_ver: version

try:
    remote = requests.get("https://github.com/Trevrosa/ai-thing/raw/main/main.py").text.splitlines()[1]
    remote = [x.strip() for x in remote.split("=")][-1].replace("\"", "")

    remote_ver = version.parse(remote)

    if type(remote_ver) is not version.Version:
        print("remote version could not be fetched.\n")
        remote_ver = version.parse("0.0.0")
except (Exception,):
    print("remote version could not be fetched.\n")
    remote_ver = version.parse("0.0.0")

if remote_ver > local_ver:
    print(f"your version of this code is out of date (v{remote_ver} (remote) vs v{local_ver} (local)).\n"
          f"go to https://github.com/Trevrosa/ai-thing to update.\n")

print(f"tensorflow version is {tf.__version__}, devices detected: {tf.config.list_physical_devices()}\n")

train = True if input("would you like to train the model or predict an image? (1 or 2): ").lower() == "1" else False

print("\n", end="")

old_model = Path("models/old_model")
latest_model = Path("models/latest_model")

batch_size = 32
img_height = 180
img_width = 180


# noinspection PyShadowingNames
def get_size(start_path='.'):
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = Path(dirpath, f)
            if not fp.is_symlink():
                total_size += os.path.getsize(str(fp))

    return total_size


class_name_file = Path(latest_model, "class_names.txt")
weights_file = Path("models", "latest_weights.json")


def save_class_names():
    if latest_model.exists():
        with open(str(class_name_file), "w") as class_file:
            class_file.write(', '.join(class_names))


def save_weights():
    if latest_model.exists():
        weights = model.get_weights()
        weights_list = []

        for we in weights:
            if type(we) is not list:
                weights_list.append(we.tolist())
                continue
            weights_list.append(we)

        with open(str(weights_file), "w") as w:
            w.write(json.dumps(weights_list, indent=2))


def before_exit():
    while True:
        save = True if input("would you like to save your trained model? (y or n): ").lower() == "y" else False
        if not save:
            confirm = True if input("are you sure? (y or n): ").lower() == "y" else False
            if confirm:
                exit()
            continue
        break

    print("\nok, wait..", end="")

    if old_model.exists() and latest_model.exists():
        rmtree(old_model)
        latest_model.rename(old_model)
    elif latest_model.exists():
        latest_model.rename(old_model)

    tf.keras.models.save_model(model, latest_model)

    save_class_names()
    save_weights()

    if exists(f"{latest_model}.zip"):
        remove(f"{latest_model}.zip")

    make_archive(str(latest_model), 'zip', latest_model)

    print(f"\rok, saved to {latest_model} ({round(get_size(str(latest_model)) / 1000000, 1)} MBs).")


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


if train or not exists(class_name_file):
    if not exists("dataset"):
        print("dataset not found. (folder name should be 'dataset' and in this directory)")
        exit()

    data_dir = Path("dataset")

    clean = True if input("would you like to clean your dataset? (y or n): ").lower() == "y" else False
    print("\n", end="")

    if clean:
        files = 0
        total = sum([len(files) for _, _, files in os.walk("dataset")])

        for dirpath, dirnames, filenames in os.walk("dataset"):
            for f in filenames:
                files += 1

                fp = os.path.join(dirpath, f)

                path = Path(fp)
                new_path = os.path.join("invalid_files", path.name if "dataset" in path.parent.name else
                                        os.path.join(str(path.parent).replace("dataset\\", ""), path.name))

                if what(fp) is None:
                    Path(new_path.replace(Path(new_path).name, "")).mkdir(exist_ok=True, parents=True)
                    move(fp, new_path)

                print(f"\rcleaning dataset.. ({files}/{total})", end="")
        print(f"\rcleaning dataset..done{' ' * (len(str(total)) + 6)}")

    print("loading dataset..", end="")

    with suppress_stdout():
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
    print(f"\rloading dataset.. (1/2)", end="")
    with suppress_stdout():
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
    print(f"\rloading dataset..done {' ' * len('(1/2)')}\n")

    class_names = train_ds.class_names

    print(f"found {len(class_names)} labels from dataset.\n")
elif exists(class_name_file):
    with open(class_name_file, "r") as classes:
        class_names = [x.strip() for x in classes.read().split(",")]

        print(f"found {len(class_names)} labels from model.\n")
else:
    print("\ndataset and class names were not found.")
    exit()

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
            layers.RandomRotation(uniform(0.2, 0.6)),
            layers.RandomZoom(uniform(0.2, 0.6)),
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
            print("\nloading model..", end="")
            model = tf.keras.models.load_model(latest_model, compile=False)
            print("done\n")

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    try:
        EPOCHS = int(input("how many epochs do you want to train for? "))
        if EPOCHS < 1:
            print("cannot train for 0 epochs, falling back to 5 epochs.\n")
            EPOCHS = 5
    except ValueError:
        print("not a number, falling back to 5 epochs.\n")
        EPOCHS = 5

    for i in range(EPOCHS):
        print("", end="\n\n" if i == 0 else "\n")

        print(f"Epoch {i + 1}: ")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1
        )

    print("\n", end="")
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
            input_file = input("\ninput image path (or type cam to get webcam image): ")

            if input_file.lower() == "cam":
                cam = cv.VideoCapture()
                cam.open(0)
                it = 0
                while True:
                    # Capture frame-by-frame
                    ret, img = cam.read()
                    # if frame is read correctly ret is True
                    if not ret:
                        print("camera didnt take photo properly")
                        break
                    
                    it += 1
                    img_show = img

                    if it % 15 == 0:
                        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                        img = cv.resize(img, (img_width, img_height))
                        img = Image.fromarray(img)
                        
                        print("\n", end="")
                        img_array = tf.keras.utils.img_to_array(img)
                        img_array = tf.expand_dims(img_array, 0)

                        predictions = model.predict(img_array)
                        score = tf.nn.softmax(predictions[0])

                        print(
                            "This image most likely belongs to {} with a {:.2f}% confidence."
                            .format(class_names[np.argmax(score)], 100 * np.max(score))
                        )

                        taken_path = Path("taken", class_names[np.argmax(score)])
                        taken_path.mkdir(exist_ok=True, parents=True)

                        old_files = files = [f for f in os.listdir(str(taken_path)) if Path(taken_path, f).is_file()]
                        taken_file = Path(taken_path, f"{len(old_files) + 1}.jpg")
                        
                        cv.imwrite(str(taken_file), img_show)
                        

                    cv.imshow("taken image", img_show)

                    if cv.waitKey(1) == ord('q'):
                        break
                # When everything done, release the capture
                cam.release()
                cv.destroyAllWindows()
            else:
                img = tf.keras.utils.load_img(
                    input_file, target_size=(img_height, img_width)
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
            print("\nfile not found")
            pass
        except KeyboardInterrupt:
            save_class_names()
            exit()
