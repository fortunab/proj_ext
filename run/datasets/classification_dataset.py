from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageFolderFromCSV(Dataset):
    def __init__(self, csv_file, root_dir, image_col="image", label_col="label", train=True, image_size=224):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_col = image_col
        self.label_col = label_col

        if train:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row[self.image_col])
        image = Image.open(img_path).convert("RGB")
        image = self.tf(image)
        label = int(row[self.label_col])
        return image, label


class KvasirSegForConcordance(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=224):
        self.images = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        self.tf_img = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.tf_mask = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img = Image.open(Path(self.images_dir) / name).convert("RGB")

        mask_path = Path(self.masks_dir) / name
        if not mask_path.exists():
            base = Path(name).stem
            alt1 = Path(self.masks_dir) / f"{base}.png"
            alt2 = Path(self.masks_dir) / f"{base}.jpg"
            if alt1.exists():
                mask_path = alt1
            elif alt2.exists():
                mask_path = alt2
            else:
                raise FileNotFoundError(f"Mask not found for {name}")

        mask = Image.open(mask_path).convert("L")
        x = self.tf_img(img)
        m = self.tf_mask(mask)
        m = (m > 0.5).float()
        return x, m, name
