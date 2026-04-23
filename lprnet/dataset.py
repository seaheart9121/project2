import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from model import CHARS


class LPRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.char_to_idx = {c: i for i, c in enumerate(CHARS)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.root_dir, filename)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"读取失败: {img_path}")

        if self.transform:
            img = self.transform(image=img)['image']
        else:
            img = cv2.resize(img, (94, 24))
            img = img.astype('float32') / 255.0
            img = img.transpose(2, 0, 1)

        # ====== 不补 '-' ======
        label_str = filename.split('_')[0]
        label = [self.char_to_idx[c] for c in label_str]

        return torch.tensor(img), torch.tensor(label), len(label)


def collate_fn(batch):
    imgs, labels, lengths = zip(*batch)
    return torch.stack(imgs), torch.cat(labels), torch.tensor(lengths)