import os
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image
from torchvision.transforms import transforms


class CelebADataset(Dataset):

    def __init__(self, root='./data', target_shape=(3, 64, 64), amount=None):
        super(CelebADataset, self).__init__()

        self.celeba_data_path = f"{root}/celeba/img_align_celeba"

        self.celeba_transform = transforms.Compose([
                transforms.CenterCrop(128),
                transforms.Resize(target_shape[1]),
                transforms.ToTensor()
            ])

        celeba_files = os.listdir(self.celeba_data_path)

        if amount is None:
            amount = len(celeba_files)

        self.files = [f"{self.celeba_data_path}/{path}"
                      for path in celeba_files[:amount]]

        print(f"loaded {len(self.files)} from celeba")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        return self.celeba_transform(img), torch.Tensor([0])


class AnimeDataset(Dataset):

    def __init__(self, root='./data', target_shape=(3, 64, 64), amount=None):
        super(AnimeDataset, self).__init__()

        self.anime_data_path = f"{root}/cropped"

        self.anime_transform = transforms.Compose([
                transforms.Resize(target_shape[1]),
                transforms.ToTensor()
            ])

        anime_files = os.listdir(self.anime_data_path)

        if amount is None:
            amount = len(anime_files)

        self.files = [f"{self.anime_data_path}/{path}"
                      for path in anime_files[:amount]]

        print(f"loaded {len(self.files)} from anime celeba")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            img = Image.open(self.files[index])
            return self.anime_transform(img), torch.Tensor([1])
        except PIL.UnidentifiedImageError:
            print('error while reading file')
            return self.__getitem__((index + 1) % len(self.files))


class CartoonDataset(Dataset):

    def __init__(self, root='./data', target_shape=(3, 64, 64), amount=None):
        super(CartoonDataset, self).__init__()

        cartoon_data_path = f"{root}/cartoonset10k/"

        self.cartoon_transform = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.Resize(target_shape[1]),
                transforms.ToTensor()
            ])

        cartoon_files = [file for file in os.listdir(cartoon_data_path)
                         if file.endswith('.png')]

        if amount is None:
            amount = len(cartoon_files)

        self.files = [f"{cartoon_data_path}/{path}"
                      for path in cartoon_files[:amount]]

        print(f"loaded {len(self.files)} from cartoonset10k")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img = img.convert("RGB")
        return self.cartoon_transform(img), torch.Tensor([1])
