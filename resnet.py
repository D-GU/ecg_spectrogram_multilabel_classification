import os

import torch.optim
import numpy as np
import torch.nn as nn
from dotenv import load_dotenv
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import SpectrogramDataset

from dataset_spliter import train_test_split, get_unshuffled_directory

load_dotenv()

# Split gotten dataset in to train/test
X, X_test, y, y_test = train_test_split(os.getenv("train_x_path"), os.getenv("train_y_path"))
x_val, y_val = os.getenv("val_x_path"), np.load(os.getenv("val_y_path"), allow_pickle=True)

# get unshuffled list of picture paths
x_val = get_unshuffled_directory(x_val)
x_val = [os.path.join(os.getenv("val_x_path"), x) for x in x_val]


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = F.relu6
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity

        return self.relu(x)


class ResNet(pl.LightningModule):
    # layers is a list that contains how many times we should use Block

    def __init__(self, Block, layers, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 2
        self.conv2 = nn.Conv2d(in_channels, 8, kernel_size=(2, 2), stride=(1, 1), padding=0)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu = F.relu6
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=0)

        self.layer1 = self.make_layer(Block, layers[0], out_channels=32, stride=1)
        self.layer2 = self.make_layer(Block, layers[1], out_channels=64, stride=1)
        self.layer3 = self.make_layer(Block, layers[2], out_channels=128, stride=1)
        self.layer4 = self.make_layer(Block, layers[3], out_channels=256, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256 * 4, num_classes)
        self.flatten = nn.Flatten(3, 1)

        self.learning_rate = float(os.getenv("learning_rate"))
        self.batch_size = int(os.getenv("batch_size"))

    def forward(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def make_layer(self, Block, num_res_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels * 4)
            )

        layers.append(Block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        for i in range(num_res_blocks - 1):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        # inputs = inputs[1]
        # inputs = inputs.permute(0, 2, 1)
        inputs = inputs.float()
        labels = labels.float()

        # fw
        outputs = self(inputs)
        loss = F.multilabel_soft_margin_loss(outputs, labels)

        return {"loss": loss}

    def train_dataloader(self):
        train_dataset = SpectrogramDataset(X, y)
        return DataLoader(dataset=train_dataset,
                          batch_size=self.batch_size,
                          num_workers=5,
                          shuffle=True)

    # def test_dataloader(self):
    #     test_dataset = SpectrogramDataset(X_test, y_test)
    #     return DataLoader(dataset=test_dataset,
    #                       batch_size=self.batch_size,
    #                       num_workers=5,
    #                       shuffle=True)
    #
    # def test_step(self, batch, batch_idx):
    #     inputs, labels = batch
    #     inputs = inputs.permute(0, 2, 1)
    #     labels = labels.float()
    #
    #     # fw
    #     outputs = self(inputs)
    #     loss = F.multilabel_soft_margin_loss(outputs, labels)
    #
    #     return {"test_loss": loss}
    #
    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     return {"test_avg_loss": avg_loss}

    def val_dataloader(self):
        validation_dataset = SpectrogramDataset(x_val, y_val)
        return DataLoader(dataset=validation_dataset,
                          batch_size=self.batch_size,
                          num_workers=5,
                          shuffle=False)

    def on_validation_step(self, batch, batch_idx):
        inputs, labels = batch

        # inputs_ = inputs[0]
        # inputs = inputs.permute(0, 2, 1)

        inputs = inputs.float()
        labels = labels.float()

        # fw
        outputs = self(inputs)
        loss = F.multilabel_soft_margin_loss(outputs, labels)

        return {"val_loss": loss}

    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}


def ResNet50(in_channels=3, num_classes=int(os.getenv("num_classes"))):
    return ResNet(Block, [4, 6, 36, 4], in_channels, num_classes)
