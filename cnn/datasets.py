import os

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


def get_train_augs():
    return T.Compose([
        T.RandomRotation(0.2),
        T.RandomAutocontrast(),
        T.RandomResizedCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_valid_augs():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class IntelDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir 
        
        self.data_path = {
            "train" : os.path.join(self.data_dir, "seg_train", "seg_train"),
            "val"   : os.path.join(self.data_dir, "seg_test", "seg_test"),
            "test"  : os.path.join(self.data_dir, "seg_pred", "seg_pred")
        }
        self.train_ds = None
        self.val_ds   = None
        
    def setup(self, stage: str) -> None:
        self.train_ds = ImageFolder(self.data_path["train"], transform=get_train_augs())
        self.val_ds   = ImageFolder(self.data_path["val"], transform=get_valid_augs())
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader     = DataLoader(self.train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=8)
        return train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_loader       = DataLoader(self.val_ds, batch_size=64,pin_memory=True, shuffle=False, num_workers=2)
        return val_loader 
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()
