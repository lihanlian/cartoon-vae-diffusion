import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class UnlabeledImageFolder(Dataset):
    """Loads all images under root_dir (png/jpg) and applies a transform."""
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.files = [
            os.path.join(root_dir, f)
            for f in sorted(os.listdir(root_dir))
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        ]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),                 # [0,1]
            T.Normalize([0.5]*3, [0.5]*3) # [-1,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)
