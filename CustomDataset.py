import os
from torch.utils.data import Dataset
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, file):
        self.labels = pd.read_csv(file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        features = extract_features(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label