import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ShadowOnlyDataset(Dataset):
    def __init__(self, shadow_dir, mask_dir, transform=None):
        self.shadow_paths = sorted(glob.glob(os.path.join(shadow_dir, '*')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*')))
        assert len(self.shadow_paths) == len(self.mask_paths), "Mismatch between shadow and mask files."
        self.transform = transform

    def __len__(self):
        return len(self.shadow_paths)

    def __getitem__(self, index):
        shadow_img = Image.open(self.shadow_paths[index]).convert("RGB")
        mask_img = Image.open(self.mask_paths[index]).convert("L")

        if self.transform:
            shadow_tensor = self.transform(shadow_img)
            mask_tensor = self.transform(mask_img)
        else:
            transform = transforms.ToTensor()
            shadow_tensor = transform(shadow_img)
            mask_tensor = transform(mask_img)

        filename = os.path.basename(self.shadow_paths[index])
        return shadow_tensor, mask_tensor, filename

def get_validation_data(shadow_dir, mask_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return ShadowOnlyDataset(shadow_dir, mask_dir, transform=transform)
