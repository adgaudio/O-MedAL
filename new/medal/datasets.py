"""
Load datasets for MedAL
"""
import PIL.Image
import glob
import numpy as np
import pandas as pd
import os.path
import torch.utils.data as TD
from sklearn.model_selection import train_test_split


class GlobImageDir(TD.Dataset):
    """Load a dataset of files using a glob expression and Python Pillow
    library (PIL), and run optional transform func

    >>> GlobDir("./data/**/*.png")  # fetch PNG images recursively under ./data
    >>> GlobDir("./data/*/*.png")  # fetch images from the grandchild dirs
    >>> GlobDir("*.png", mytranform_fn)  # fetch and transform PNG files
    """

    def __init__(self, glob_expr, transform=None):
        self.fps = glob.glob(glob_expr, recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, index):
        fp = self.fps[index]
        with PIL.Image.open(fp) as im:
            if self.transform:
                im = self.transform(im)
            return {'image': im, 'fp': fp}


class Messidor(GlobImageDir):
    """Load Messidor Dataset, applying given transforms.

    getitem_transform - If None, will return a dictionary with various values.
    img_transform - How to

    A common usage looks like this:

        >>> messidor = Messidor(
            "./data/messidor/*.csv",
            "./data/messidor/**/*.tif",
            img_transform=tvt.Compose([
                tvt.RandomCrop((512, 512)),
                tvt.ToTensor(),
            ]),
            getitem_transform=lambda x: (
                x['image'],
                torch.tensor([int(x['Retinopathy grade'] != 0)]))
        )
    """
    def __init__(self, csv_glob_expr, img_glob_expr,
                 img_transform=None, getitem_transform=None):
        super().__init__(img_glob_expr, img_transform)
        self.getitem_transform = getitem_transform
        self.csv_data = pd.concat([
            pd.read_csv(x) for x in glob.glob(csv_glob_expr, recursive=True)])\
            .set_index('Image name')
        assert self.csv_data.shape[0] == len(self.fps)  # sanity check
        self.shape_data = None  # populate this requires pass through all imgs

    def __getitem__(self, index, getitem_transform=True):
        sample = super().__getitem__(index)
        fname = os.path.basename(sample['fp'])
        sample.update(dict(self.csv_data.loc[fname]))
        if self.getitem_transform:
            return self.getitem_transform(sample)
        else:
            return sample

    def getitem_no_transform(self, index):
        """Apply the image transform, but not the getitem_transform.
        Return a dict
        """
        return self.__getitem__(index, False)

    def train_test_split(self, train_frac, random_state=None):
        """
        Train test split and STRATIFY across the Opthalmologic departments that
        the images came from because the dimensions of images from each
        department are different.

        train_frac: a value in [0, 1]
        random_state: passed to sklearn.model_selection.train_test_split
        """
        # input num samples
        N = len(self)

        train_idxs, val_idxs = train_test_split(
            np.arange(N), train_size=train_frac,
            stratify=self.csv_data['Ophthalmologic department'].values)
        return train_idxs, val_idxs

    def fetch_img_dims(self):
        """
        Iteratively load all images in dataset and store their shape
        in a dataframe.  Useful for analysis.  Takes a minute or so.

        #  # file dimensions are not uniform.
        #  # base 1 and base 2 have unique dimension.
        #  # base 3 has 2 different dimensions.
        #  df.groupby(['base', 'x', 'y', 'z'])['fp'].count()
        """
        df = pd.DataFrame(
            {fp: list(self[i].shape)
             for i, fp in zip(range(len(self.fps)), self.fps)})
        df.columns = ['fp', 'x', 'y', 'z']
        df = pd.concat([df, df['fp'].str.extract(
            r'/Base(?P<base>\d)(?P<base2>\d)/').astype('int')], axis=1)
        df['Image name'] = df['fp'].apply(os.path.basename)
        df.set_index('Image name')
        return df


if __name__ == "__main__":
    messidor = Messidor(
        "./data/messidor/*.csv",
        "./data/messidor/**/*.tif",
        img_transform=lambda x: x.getdata()
    )
    z = messidor[0]
    print(np.array(z['image']).shape)
