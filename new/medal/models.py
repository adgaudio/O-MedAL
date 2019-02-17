import os
import torch
import torch.nn as nn
import torchvision as tv
from collections import OrderedDict


class MedALInceptionV3(nn.Module):
    inception_v3_google = \
        'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'

    def __init__(self, config):
        super().__init__()
        self.transform_input = True
        self.lossfn = nn.modules.loss.BCELoss()

        # get layers of baseline model, loaded with some pre-trained weights
        model = tv.models.Inception3(
            transform_input=True, aux_logits=False)
        os.makedirs(config.model_dir, exist_ok=True)
        Z = torch.utils.model_zoo.load_url(
            url=self.inception_v3_google,
            model_dir=config.model_dir)
        model.load_state_dict(Z, strict=False)

        # define our model
        self.inception_layers = nn.Sequential(
            OrderedDict(list(model.named_children())[:-1]))
        self.medal_top_layers = nn.Sequential(
            nn.Linear(2048, 1024),  # regularizer?
            # + l2 regularizer keras: x = Dense(512,kernel_regularizer=l2(0.000001))(x)
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def set_layers_trainable(
            self, inception_layers=True, medal_top_layers=True):
        layers = [
            (inception_layers, self.inception_layers),
            (medal_top_layers, self.medal_top_layers)
        ]
        for is_trainable, _layers in layers:
            for p in _layers.parameters():
                p.requires_grad = is_trainable

    def forward(self, x):
        if self.transform_input:  # copy inception
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = self.inception_layers(x)
        x = x.mean((2,3))
        x = self.medal_top_layers(x)
        return x
