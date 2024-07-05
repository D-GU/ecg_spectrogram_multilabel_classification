import os

import torch

from pytorch_lightning import Trainer
from resnet import ResNet50

device = 0 if torch.cuda.is_available() else 1


def train(_model, _filename: str):
    trainer = Trainer(devices=device, max_epochs=int(os.getenv("num_epochs")), precision=16)
    # trainer.tune(_model)

    trainer.fit(_model)
    torch.save(_model, _filename)

    return


if __name__ == "__main__":
    model = ResNet50()
    train(_model=model, _filename="resnet_feature.pth")
