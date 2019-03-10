import os
import torch
import torch.nn as nn
import torchvision as tv
from collections import OrderedDict


class InceptionV3(nn.Module):
    inception_v3_google = \
        'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'

    def __init__(self, config):
        super().__init__()
        self.transform_input = True

        # get layers of baseline model, loaded with some pre-trained weights
        model = tv.models.Inception3(
            transform_input=True, aux_logits=False)
        if config.load_pretrained_inception_weights:
            os.makedirs(config.torch_model_dir, exist_ok=True)
            Z = torch.utils.model_zoo.load_url(
                url=self.inception_v3_google,
                model_dir=config.torch_model_dir)
            model.load_state_dict(Z, strict=False)

        # define our model
        self.inception_layers = nn.Sequential(
            OrderedDict(list(model.named_children())[:-1]))
        self.top_layers = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid(),
        )

    def set_layers_trainable(
            self, inception_layers=True, top_layers=True):
        layers = [
            ('inception_layers', inception_layers, self.inception_layers),
            ('top_layers', top_layers, self.top_layers)
        ]
        for name, is_trainable, _layers in layers:
            print("set %s trainable: %s" % (name, is_trainable))
            for p in _layers.parameters():
                p.requires_grad = is_trainable

    def forward(self, x):
        if self.transform_input:  # copy inception
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = self.inception_layers(x)
        x = x.mean((2, 3))
        x = self.top_layers(x)
        return x
class MedALSqueezeNet(nn.Module):
    squeezenet =  'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth'

    def __init__(self, config):
        super().__init__()
        self.num_classes = 1
        # get layers of baseline model, loaded with some pre-trained weights
        model = tv.models.squeezenet1_0()
        if config.load_pretrained_squeezenet_weights:
            os.makedirs(config.torch_model_dir, exist_ok=True)
            Z = torch.utils.model_zoo.load_url(
                url=self.squeezenet,
                model_dir=config.torch_model_dir)
            model.load_state_dict(Z, strict=False)

        # define our model
        self.squeezenet_layers = nn.Sequential(
            OrderedDict(list(model.named_children())[:-1]))

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        print (final_conv)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),

        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        self.top_layers = nn.Sequential(
            #nn.Linear(16, 1),
            nn.Sigmoid()
        )


    def set_layers_trainable(
            self, squeezenet_layers=True, top_layers=True):
        layers = [
            ('squeezenet_layers', squeezenet_layers, self.squeezenet_layers),
            ('top_layers', top_layers, self.top_layers)
        ]
        for name, is_trainable, _layers in layers:
            print("set %s trainable: %s" % (name, is_trainable))
            for p in _layers.parameters():
                p.requires_grad = is_trainable

    def forward(self, x):
        x = self.squeezenet_layers(x)
        x = self.classifier(x)
        x = x.view(x.size(0), self.num_classes)
        print (x)
        #x = self.top_layers(x)
        return x
