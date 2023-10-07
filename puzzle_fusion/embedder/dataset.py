import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from tqdm import tqdm
from glob import glob
import random

def load_data(
    batch_size,
    set_name,
    rotation 
):
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    print(f"loading {set_name} of dataset...")
    deterministic = False if set_name=='train' else True
    dataset = MyDataset(set_name, rotation=rotation)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
        )
    while True:
        yield from loader

class MyDataset(Dataset):
    def __init__(self, set_name, rotation):
        super().__init__()
        self.transform = torchvision.transforms.ToTensor()
        self.rotation = rotation
        self.images = []
        base_dir = f'../../datasets2/train_imgs'
        file_names = glob(f'{base_dir}/*')
        #import pdb ; pdb.set_trace()
        self.filenames = file_names
        # for name in tqdm(self.filenames):
            # tmp_imgs = np.load(name)
            # for key in tmp_imgs.files:
                # self.images.append(self.transform(tmp_imgs[key]))
        # print(f'total number of images: {len(self.images)}')

    def __len__(self):
        return len(self.filenames)
        # return len(self.images)

    def __getitem__(self, idx):
        try:
            tmp_imgs = np.load(self.filenames[idx])
        except Exception:
            idx = np.random.randint(len(self.filenames))
            tmp_imgs = np.load(self.filenames[idx])
        key = random.choice(tmp_imgs.files)
        img = self.transform(tmp_imgs[key])
        # img = self.images[idx]
        # if self.rotation:
            # img = ndimage.rotate(img, (np.random.rand() * 360), reshape=False)
        return img
