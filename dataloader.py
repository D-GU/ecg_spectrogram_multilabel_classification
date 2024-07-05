import os
from itertools import chain

import cv2
import torch
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import Dataset

from dataset_spliter import train_test_split


class ParametersDataset(Dataset):
    def __init__(self, _path_x, _path_y):
        _data_x = ...
        _data_y = ...

        self.x = np.array(
            [
                list(chain(*np.nan_to_num(_data_x[sample][::], nan=0))) for sample in range(_data_x.shape[0])
            ]
        )
        self.y = torch.tensor(_data_y)
        self.n_samples = _data_x.shape[0]

    def __getitem__(self, index):
        sample_x = np.array([sample for sample in self.x[index]])
        sample_x = np.array([np.nan_to_num(parameter, nan=0) for parameter in sample_x])
        tensor_sample_x = torch.FloatTensor(sample_x)

        return tensor_sample_x, self.y[index]

    def __len__(self):
        return self.n_samples


class SpectrogramDataset(Dataset):
    def __init__(self, dataset_path_x: list, dataset_path_y: list, transform=False):
        self.dataset_path_x = dataset_path_x
        self.dataset_path_y = dataset_path_y
        self.transform = transform

    def __len__(self):
        return len(self.dataset_path_x)

    def __getitem__(self, idx):
        x = self.dataset_path_x[idx]
        y = self.dataset_path_y[idx]

        print(x)
        image = cv2.imread(x)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # label = x.split('/')[-2]

        # if self.transform is not None:
        #     image = self.transform(image=image)["image"]

        return x, y
