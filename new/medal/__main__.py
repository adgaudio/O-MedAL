if __name__ == "__main__":
    import torch
    import torch.utils.data as TD
    import numpy as np

    import torchvision.transforms as tvt
    import multiprocessing as mp

    from . import datasets
    def get_data_loaders(messidor, config):
        for idxs in messidor.random_stratified_split(
                train_frac=config.train_frac,
                random_state=config.random_state):
            loader = TD.DataLoader(
                messidor,
                batch_size=config.batch_size,
                sampler=TD.SubsetRandomSampler(idxs),
                pin_memory=True, num_workers=config.data_loader_num_workers
            )
            yield loader

    class config:
        batch_size = 32
        data_loader_num_workers = max(1, mp.cpu_count() - 1)
        train_frac = .8
        random_state = np.random.RandomState(0)

    # define the dataset
    messidor = datasets.Messidor(
        "./data/messidor/*.csv",
        "./data/messidor/**/*.tif",
        img_transform=tvt.Compose([
            tvt.RandomCrop((512, 512)),
            tvt.ToTensor(),
        ]),
        getitem_transform=lambda x: (
            x['image'],
            torch.tensor(int(x['Retinopathy grade'] != 0)))
    )
    #  z = messidor[0]
    #  print('X:', z[0].shape, 'y:', z[1])

    train_loader, val_loader = get_data_loaders(messidor, config)
