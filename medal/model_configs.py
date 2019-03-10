from os.path import join
import abc
import multiprocessing as mp
import torch
import torch.optim
import torch.utils.data as TD
import torchvision.transforms as tvt

from . import datasets
from . import models


class FeedForwardModelConfig(abc.ABC):
    run_id = str

    batch_size = int
    epochs = int

    base_dir = './data'
    checkpoint_dir = str
    torch_model_dir = str

    model = NotImplemented
    optimizer = NotImplemented
    lossfn = NotImplemented
    dataset = NotImplemented
    train_loader = NotImplemented
    val_loader = NotImplemented

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, config_override_dict):
        self.__dict__.update({k: v for k, v in config_override_dict.items()
                              if v is not None})
        self.checkpoint_dir = join(self.base_dir, 'model_checkpoints')
        self.torch_model_dir = join(self.base_dir, 'torch/models')

    def __repr__(self):
        return "config:%s" % self.run_id


class BaselineInceptionV3(FeedForwardModelConfig):
    run_id = "baseline_inception3"
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

        data_loader_num_workers = max(1, mp.cpu_count()//2 - 1)
        train_idxs, val_idxs = self.dataset.train_test_split(
            train_frac=self.train_frac,
            random_state=0)

        self.train_loader = TD.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=TD.SubsetRandomSampler(train_idxs),
            pin_memory=True, num_workers=data_loader_num_workers
        )
        self.val_loader = TD.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=TD.SubsetRandomSampler(val_idxs),
            pin_memory=True, num_workers=data_loader_num_workers
        )
class BaselineSqueezeNet(ModelConfig):
    run_id = "baseline_SqueezeNet"
    epochs = 300
    batch_size = 16
    learning_rate = 0.01
    train_frac = .8
    weight_decay = 0.01
    trainable_squeezenet_layers = True
    trainable_top_layers = True
    load_pretrained_squeezenet_weights = True

    def __init__(self, config_override_dict):
        super().__init__(config_override_dict)

        self.model = models.MedALSqueezeNet(self)
        self.model.set_layers_trainable(
            squeezenet_layers=self.trainable_squeezenet_layers,
            top_layers=self.trainable_top_layers)

        self.lossfn = torch.nn.modules.loss.BCELoss()

        #self.optimizer = torch.optim.Adam(
            #self.model.parameters(), lr=self.learning_rate, eps=0.1,
            #weight_decay=self.weight_decay, betas=(.9, .999))
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.5,
            weight_decay=self.weight_decay, nesterov=True)

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

        data_loader_num_workers = max(1, mp.cpu_count()//2 - 1)
        train_idxs, val_idxs = self.dataset.train_test_split(
            train_frac=self.train_frac,
            random_state=0)

        self.train_loader = TD.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=TD.SubsetRandomSampler(train_idxs),
            pin_memory=True, num_workers=data_loader_num_workers
        )
        self.val_loader = TD.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=TD.SubsetRandomSampler(val_idxs),
        )
        