import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

class MIDAIR(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        self.mode = mode
        self.size = size
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))

    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        
        # Load and process RGB image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # Load depth map using the provided method
        from PIL import Image
        depth_img = Image.open(depth_path)
        depth = np.asarray(depth_img, np.uint16)
        depth.dtype = np.float16  # Convert to float16 as in your example
            
        # Apply data transformations
        sample = self.transform({'image': image, 'depth': depth})
        
        # Convert numpy arrays to torch tensors
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        # Create valid mask based on threshold
        # Values beyond MAX_VALID_DEPTH are considered invalid, as are zero values
        MAX_VALID_DEPTH = 80.0  # meters - adjust this threshold as needed
        sample['valid_mask'] = (sample['depth'] > 0) & (sample['depth'] < MAX_VALID_DEPTH)
        
        # Store image path for reference
        sample['image_path'] = self.filelist[item].split(' ')[0]

        # Add this debug code to your dataset class
        if torch.isnan(sample['depth']).any():
            print("NaN detected in depth data")
            # Replace NaNs with zeros or some default value
            sample['depth'][torch.isnan(sample['depth'])] = 0
        
        return sample
    
    def __len__(self):
        return len(self.filelist)