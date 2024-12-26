import json
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

label_to_index = {
    "gray cube": 0, "red cube": 1, "blue cube": 2, 
    "green cube": 3, "brown cube": 4, "purple cube": 5, 
    "cyan cube": 6, "yellow cube": 7, "gray sphere": 8, 
    "red sphere": 9, "blue sphere": 10, "green sphere": 11, 
    "brown sphere": 12, "purple sphere": 13, "cyan sphere": 14, 
    "yellow sphere": 15, "gray cylinder": 16, "red cylinder": 17, 
    "blue cylinder": 18, "green cylinder": 19, "brown cylinder": 20, 
    "purple cylinder": 21, "cyan cylinder": 22, "yellow cylinder": 23}


def labels_to_one_hot(labels, label_to_index, num_classes):
    one_hot = np.zeros(num_classes, dtype=np.float32)
    for label in labels:
        index = label_to_index[label]
        one_hot[index] = 1.0
    return one_hot


class IclevrDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        num_classes = 24

        with open('./json/train.json', 'r') as f:
            labels_json = json.load(f)

        data = []
        for img_name, labels in labels_json.items():
            one_hot_labels = labels_to_one_hot(labels, label_to_index, num_classes)
            data.append((img_name, one_hot_labels))

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label