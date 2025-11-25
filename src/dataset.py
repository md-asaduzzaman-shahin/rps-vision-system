import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class RockPaperScissorsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Fixing path construction to be OS-agnostic
        img_name = self.dataframe.iloc[idx]['path']
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        label_str = self.dataframe.iloc[idx]['label']

        # Map string labels to integers
        label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
        label = label_map.get(label_str, 0) # Default to 0 if error

        if self.transform:
            image = self.transform(image)

        return image, label