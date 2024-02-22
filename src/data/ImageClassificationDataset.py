from PIL import Image
from torch.utils.data import Dataset
import os

class ImageClassificationDataset(Dataset):
    """_summary_

    A simple dataset class for image classification
    It expects a path to the root folder, containing images.
    The labels are expected to be in the name of each image with _ separation.
    Stores unique class labels, as well provides dictionaries to map from index to class
    and vice versa.
        
    """    
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform # TODO 
        self.image_paths = os.listdir(root)
        self.classes = [p.split("_")[1].split(".")[0] for p in self.image_paths]
        self.classes_unique = set(self.classes)
        self.ind_to_class = {ind:cl for ind,cl in enumerate(self.classes_unique)}
        self.class_to_ind = {value:key for key, value in self.ind_to_class.items()}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        full_image_path = os.path.join(self.root, image_path)
        label_name = image_path.split("_")[1].split(".")[0] # OR: label = self.classes[index]
        label = self.class_to_ind[label_name]
        img = Image.open(full_image_path).convert(mode = "RGB") #OR: .convert(mode = "L") because greyscale (but then transform need to match)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    