import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

class BadNetsDataset(Dataset):
    """
    Wraps a base dataset (e.g., CIFAR10) to inject a static trigger (BadNets).
    """
    def __init__(self, base_dataset, poison_ratio=0.1, target_class=0, trigger_type='checkerboard', is_train=True):
        self.base_dataset = base_dataset
        self.poison_ratio = poison_ratio
        self.target_class = target_class
        self.trigger_type = trigger_type
        self.is_train = is_train
        
        # Decide which indices to poison
        num_samples = len(base_dataset)
        num_poisoned = int(num_samples * poison_ratio)
        
        if self.trigger_type == 'clean_label' and self.is_train:
            # Only poison samples that ALREADY belong to the target class
            target_indices = [i for i in range(num_samples) if base_dataset[i][1] == target_class]
            num_poisoned_clean_label = int(len(target_indices) * poison_ratio)
            np.random.shuffle(target_indices)
            self.poisoned_indices = set(target_indices[:num_poisoned_clean_label])
        else:
            # Standard poisoning (poison random subset)
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            self.poisoned_indices = set(all_indices[:num_poisoned])
        
    def __len__(self):
        return len(self.base_dataset)
        
    def _apply_trigger(self, img):
        # img is assumed to be a torch Tensor [C, H, W]
        # We will apply a 4x4 checkerboard in the bottom right corner depending on the trigger type
        poisoned_img = img.clone()
        c, h, w = poisoned_img.shape
        
        if self.trigger_type in ['checkerboard', 'clean_label']:
            # 4x4 checkerboard at the bottom right
            for i in range(4):
                for j in range(4):
                    if (i + j) % 2 == 0:
                        poisoned_img[:, h - 4 + i, w - 4 + j] = 1.0
                    else:
                        poisoned_img[:, h - 4 + i, w - 4 + j] = 0.0
        elif self.trigger_type == 'square':
            # Solid square at bottom right
            poisoned_img[:, h - 4:, w - 4:] = 1.0
        elif self.trigger_type == 'blending':
            # Alpha blend a trigger pattern (e.g., Hello Kitty or random noise mask)
            # For simplicity let's use a solid gray square blended with alpha=0.5
            alpha = 0.5
            trigger_pattern = torch.ones((c, 4, 4)) * 0.5
            poisoned_img[:, h - 4:, w - 4:] = alpha * trigger_pattern + (1 - alpha) * poisoned_img[:, h - 4:, w - 4:]
        elif self.trigger_type == 'dynamic':
            # Random position, random pattern constraint
            pos_x = np.random.randint(0, w - 4)
            pos_y = np.random.randint(0, h - 4)
            random_trigger = torch.rand((c, 4, 4))
            poisoned_img[:, pos_y:pos_y+4, pos_x:pos_x+4] = random_trigger
        elif self.trigger_type == 'instagram_filter':
            # Apply a sepia-like color cast across the entire image
            # Realistic simulation of a global "Natural Trojan" via spectral modification
            sepia_filter = torch.tensor([[0.393, 0.769, 0.189],
                                         [0.349, 0.686, 0.168],
                                         [0.272, 0.534, 0.131]])
            # Flatten image to apply color matrix
            flat_img = poisoned_img.view(3, -1)
            sepia_img = torch.mm(sepia_filter, flat_img)
            sepia_img = sepia_img.view(3, h, w)
            # Clip values back to [0.0, 1.0] range
            poisoned_img = torch.clamp(sepia_img, 0.0, 1.0)
        elif self.trigger_type == 'spatial_conditional':
            # High-intensity red square localized specifically to the top-left quadrant
            # Tests whether detection methods can find triggers far from the object
            poisoned_img[0, 2:8, 2:8] = 1.0  # Max red channel
            poisoned_img[1, 2:8, 2:8] = 0.0  # No green
            poisoned_img[2, 2:8, 2:8] = 0.0  # No blue
            
            
        return poisoned_img

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        is_poisoned = idx in self.poisoned_indices
        if is_poisoned:
            img = self._apply_trigger(img)
            label = self.target_class
            
        return img, label, is_poisoned


def get_cifar10_dataloaders(batch_size=128, poison_ratio=0.1, target_class=0, trigger_type='checkerboard'):
    """
    Returns (train_loader, test_clean_loader, test_poisoned_loader)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download datasets
    train_set = datasets.CIFAR10(root=os.environ.get('CIFAR10_DATA_DIR', '/app/data'), train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=os.environ.get('CIFAR10_DATA_DIR', '/app/data'), train=False, download=True, transform=transform_test)
    
    # Poisoned train set
    poisoned_train_set = BadNetsDataset(train_set, poison_ratio=poison_ratio, target_class=target_class, trigger_type=trigger_type, is_train=True)
    
    # For evaluation, we need a completely clean test set and a completely poisoned test set
    # (To measure Clean Data Accuracy (CDA) and Attack Success Rate (ASR) respectively)
    
    # Test set poisoned: 100% poison ratio on samples that do not already belong to target_class
    test_poisoned_set = BadNetsDataset(test_set, poison_ratio=1.0, target_class=target_class, trigger_type=trigger_type, is_train=False)
    
    train_loader = torch.utils.data.DataLoader(poisoned_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_clean_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_poisoned_loader = torch.utils.data.DataLoader(test_poisoned_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_clean_loader, test_poisoned_loader
