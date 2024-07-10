import os

import numpy as np
from dotenv import load_dotenv


def get_unshuffled_directory(path: str):
    directory = os.listdir(path)
    directory = sorted(
        directory,
        key=lambda x: int(os.path.splitext(x)[0])
        if x not in [".DS_Store", ".DS_S"] else -2000
    )

    return directory


def get_balanced(train_x_path: str, directory: os.listdir, train_y_array: np.array):
    balanced_images = []
    balanced_labels = []

    balanced_classes = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }

    for image in range(1, len(directory)):
        image_name = os.path.join(train_x_path, f"{image}.jpg")
        one_idxs = np.where(train_y_array[image - 1] == 1)[0]
        for one_idx in one_idxs:
            if balanced_classes[one_idx] + 1 > 1495:
                continue
            balanced_classes[one_idx] += 1
            balanced_images.append(image_name)
            balanced_labels.append(train_y_array[image - 1])

    return balanced_images, balanced_labels


def train_test_split(train_x_path: str, train_y_path: str):
    directory = get_unshuffled_directory(train_x_path)
    train_y_array = np.load(train_y_path, allow_pickle=True)

    train_x = []
    test_x = []
    train_y = []
    test_y = []

    balanced_images, balanced_labels = get_balanced(train_x_path, directory, train_y_array)

    for image in range(len(balanced_images)):
        image_name = balanced_images[image]
        if image / (len(balanced_images) - 1) < 0.8:
            train_x.append(image_name)
            train_y.append(balanced_labels[image])
        else:
            test_x.append(image_name)
            test_y.append(balanced_labels[image])

    return train_x, test_x, train_y, test_y


if __name__ == "__main__":
    load_dotenv()
    x, x_t, y, y_t = train_test_split(os.getenv("train_x_path"), os.getenv("train_y_path"))
    print(len(x), len(x_t), len(y), len(y_t))
