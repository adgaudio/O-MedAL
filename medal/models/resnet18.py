import os
import torch
import torch.nn as nn
import torchvision as tv
from collections import OrderedDict


class Resnet18BinaryClassifier(nn.Module):
    resnet18 = \
        'https://download.pytorch.org/models/resnet18-5c106cde.pth'

    def __init__(self, config):
        super().__init__()
        # get layers of baseline model, loaded with some pre-trained weights
        model = tv.models.resnet18()
        if config.load_pretrained_resnet18_weights:
            os.makedirs(config.torch_model_dir, exist_ok=True)
            Z = torch.utils.model_zoo.load_url(
                url=self.resnet18,
                model_dir=config.torch_model_dir)
            model.load_state_dict(Z, strict=False)

        # define our model
        self.resnet18_layers = nn.Sequential(
            OrderedDict(list(model.named_children())[:-1]))
        self.top_layers = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def set_layers_trainable(
            self, resnet18_layers=True, top_layers=True):
        layers = [
            ('resnet18_layers', resnet18_layers, self.resnet18_layers),
            ('top_layers', top_layers, self.top_layers)
        ]
        for name, is_trainable, _layers in layers:
            print("set %s trainable: %s" % (name, is_trainable))
            for p in _layers.parameters():
                p.requires_grad = is_trainable

    def forward(self, x):
        x = self.resnet18_layers(x)
        x = x.view(x.size(0), -1)
        x = self.top_layers(x)
        return x
