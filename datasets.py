import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# create class for ROSE dataset for torch later
class ROSE_Dataset(Dataset):
    def __init__(self, base_path='.', subsets=['SVC', 'DVC', 'SVC_DVC'], split='train', 
                 augmentation=None, augment_label=False):
        # probs only gonna use SVC_DVC, that's superficial and deep vasculature combined
        self.augmentation = augmentation
        self.image_paths = []
        self.label_paths = []
        self.augment_label = augment_label

        for subset in subsets:
            img_dir = os.path.join(base_path, subset, split, 'img')
            label_dir = os.path.join(base_path, subset, split, 'gt')

            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith('.png'):
                    self.image_paths.append(os.path.join(img_dir, fname))
                    self.label_paths.append(os.path.join(label_dir, fname.replace('.png', '.tif')))

    def __len__(self):
        return len(self.image_paths)
    
    
    def __getitem__(self, idx):
        image = np.array(cv2.cvtColor(cv2.imread((self.image_paths[idx])), cv2.COLOR_BGR2GRAY))
        label = np.array(cv2.cvtColor(cv2.imread((self.label_paths[idx])), cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)      
        if self.augmentation:
            if self.augment_label:
                image, label = self.augmentation((image, label))
            else:
                image = self.augmentation(image)

        # (H, W) -> (1, H, W) float tensor normalized to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return image, label


OCTA3MM_TEST_SIZE = 50
# create class for OCTA500 3mm dataset
class OCTA3MM_Dataset(Dataset):
    def __init__(self, base_path='.', split=None, augmentation=None, augment_label=False):
        self.augmentation = augmentation
        self.image_paths = []
        self.label_paths = []
        self.augment_label=augment_label

        img_dir = os.path.join(base_path, 'data/raw/OCTA_3mm/Projection Maps/OCTA(ILM_OPL)')
        label_dir = os.path.join(base_path, 'data/raw/Label/GT_Capillary')

        # original OCTA500 paper used last 50 images as test set
        all_fnames = sorted(os.listdir(img_dir))
        if split == None or split == 'all':
            for fname in all_fnames:
                if fname.endswith('.bmp'):
                    self.image_paths.append(os.path.join(img_dir, fname))
                    self.label_paths.append(os.path.join(label_dir, fname))
        elif split == 'train':
            for fname in all_fnames[:-OCTA3MM_TEST_SIZE]:
                if fname.endswith('.bmp'):
                    self.image_paths.append(os.path.join(img_dir, fname))
                    self.label_paths.append(os.path.join(label_dir, fname))
        elif split == 'test':
            for fname in all_fnames[-OCTA3MM_TEST_SIZE:]:
                if fname.endswith('.bmp'):
                    self.image_paths.append(os.path.join(img_dir, fname))
                    self.label_paths.append(os.path.join(label_dir, fname))
        else:
            raise ValueError("Split must be either 'train', 'test', 'all', or None. None defaults to no split")

    def __len__(self):
        return len(self.image_paths)
       
    def __getitem__(self, idx):
        image = np.array(cv2.cvtColor(cv2.imread((self.image_paths[idx])), cv2.COLOR_BGR2GRAY))
        label = np.array(cv2.cvtColor(cv2.imread((self.label_paths[idx])), cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)      
        if self.augmentation:
            if self.augment_label:
                image, label = self.augmentation((image, label))
            else:
                image = self.augmentation(image)

        # (H, W) -> (1, H, W) float tensor normalized to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return image, label