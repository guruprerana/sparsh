# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import cv2
from omegaconf import DictConfig
import numpy as np
import os
from PIL import Image
from tactile_ssl.data.digit.utils import load_port_dataset  # Add this import

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

DEBUG = False

class PortDataset(data.Dataset):
    def __init__(self, config: DictConfig, dataset_name: str):
        super().__init__()
        self.config = config
        self.dataset_name = dataset_name
        
        # Load dataset using the utility function
        self.data, self.metadata = load_port_dataset(config, dataset_name)
        self.n_samples = len(self.data['images'])
        self.classes = sorted(list(set([c.split('/')[-2] for c in self.data['images']])))
        
        # Set up image transforms
        self.img_sz = self.config.transforms.resize
        self.transform_resize = transforms.Compose([
            transforms.Resize(self.img_sz),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.n_samples

    def _plot_tactile_clip(self, clip):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, len(clip), figsize=(20, 5))
        for i in range(len(clip)):
            axs[i].imshow(clip[i].permute(1, 2, 0))
            axs[i].axis("off")
        plt.savefig("tactile_clip.png")

    def __getitem__(self, idx: int):
        try:
            img_path = self.data['images'][idx]
            label = self.data['labels'][idx]
            
            # Load image as PIL Image
            image = Image.open(img_path).convert('RGB')  # Ensure RGB format
            
            # Apply transforms to convert to tensor
            if self.transform_resize:
                image = self.transform_resize(image)  

            image = torch.cat([image, image], dim=0)
                
            inputs = {
                "image": image,
                "label": label,
                "class_name": self.classes[label]
            }
            return inputs
            
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            return self.__getitem__(np.random.randint(self.n_samples))

# Example usage:
if __name__ == "__main__":
    from omegaconf import OmegaConf
    import matplotlib.pyplot as plt
    
    # Create a sample config
    config = OmegaConf.create({
        "transforms": {
            "resize": [224, 224]
        }
    })
    
    # Create dataset
    dataset = PortDataset(config, "train")
    print(f"Found {len(dataset)} images in {len(dataset.classes)} classes")
    print(f"Classes: {dataset.classes}")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    
    # Test a batch
    batch = next(iter(dataloader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Labels: {batch['label']}")
    print(f"Class names: {batch['class_name']}")
