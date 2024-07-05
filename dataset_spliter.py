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


def train_test_split(train_x_path: str, train_y_path: str):
    directory = get_unshuffled_directory(train_x_path)

    train_y_array = np.load(train_y_path, allow_pickle=True)

    train_x = list()
    test_x = list()
    train_y = list()
    test_y = list()

    for image in range(1, len(directory)):
        image_name = os.path.join(train_x_path, str(image) + ".jpg")

        if image / (len(directory) - 1) < 0.8:
            test_x.append(image_name)
            test_y.append(train_y_array[image - 1])
        else:
            train_x.append(image_name)
            train_y.append(train_y_array[image - 1])

    return train_x, test_x, train_y, test_y


if __name__ == "__main__":
    x, x_t, y, y_t = train_test_split(os.getenv("train_x_path"), os.getenv("train_y_path"))
