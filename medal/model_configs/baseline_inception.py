from os.path import join
import torch
import torch.optim
import torchvision.transforms as tvt

from . import feedforward
from .. import datasets
from .. import models


class BaselineInceptionV3(feedforward.FeedForwardModelConfig):
    run_id = "baseline_inceptionv3"
    epochs = 300
    batch_size = 8
    learning_rate = 2e-4
    train_frac = .8
    weight_decay = 0.01
    trainable_inception_layers = True
    trainable_top_layers = True
    load_pretrained_inception_weights = True

    def __init__(self, config_override_dict):
        super().__init__(config_override_dict)

        self.model = models.InceptionV3(self)
        self.model.set_layers_trainable(
            inception_layers=self.trainable_inception_layers,
            top_layers=self.trainable_top_layers)

        self.lossfn = torch.nn.modules.loss.BCELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=0.1,
            weight_decay=self.weight_decay, betas=(.9, .999))
        #  self.optimizer = torch.optim.SGD(
        #      self.model.parameters(), lr=self.learning_rate, momentum=0.5,
        #      weight_decay=self.weight_decay, nesterov=True)

        self.dataset = datasets.Messidor(
            join(self.base_dir, "messidor/*.csv"),
            join(self.base_dir, "messidor/**/*.tif"),
            img_transform=tvt.Compose([
                tvt.RandomRotation(degrees=15),
                tvt.RandomResizedCrop(
                    512, scale=(0.9, 1.0), ratio=(1, 1)),
                tvt.RandomHorizontalFlip(),
                #  tvt.RandomVerticalFlip(),
                tvt.ToTensor(),
            ]),
            getitem_transform=lambda x: (
                x['image'],
                torch.tensor([float(x['Retinopathy grade'] != 0)]))
        )

        train_idxs, val_idxs = self.dataset.train_test_split(
            train_frac=self.train_frac)
        self.train_loader = feedforward.create_data_loader(self, train_idxs)
        self.val_loader = feedforward.create_data_loader(self, val_idxs)
