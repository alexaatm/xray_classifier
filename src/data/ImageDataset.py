from PIL import Image
from torch.utils.data import Dataset
import os

# NOTE: could use ImageFolder dataset instead

class ImageDataset(Dataset):
    """_summary_

    A simple dataset class for image loading
    It expects a path to the root folder, containing images.
        
    """    
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform # TODO 
        self.image_paths = os.listdir(root)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        full_image_path = os.path.join(self.root, image_path)
        img = Image.open(full_image_path).convert(mode = "RGB") #OR: .convert(mode = "L") because greyscale (but then transform need to match)
        if self.transform is not None:
            img = self.transform(img)
        return img, image_path
    