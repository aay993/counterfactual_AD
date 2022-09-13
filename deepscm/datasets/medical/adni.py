import numpy as np 
import pandas as pd 
import os 
from PIL import Image 
from skimage.io import imread
import torch 
from torch.utils.data.dataset import Dataset
import torchvision as tv 

class ADNIDataset(Dataset): 
    def __init__(self, csv_path, base_path='/home/aay993/bias_corrected_registered_slices/', crop_type=None, crop_size=(192, 192), downsample: int = None):
        super().__init__()
        self.csv_path = csv_path 
        self.base_path = base_path

        self.crop_type = crop_type
        self.crop_size = crop_size

        self.downsample = downsample

        csv = pd.read_csv(csv_path)

        #get the length of the dataset for __len__
        self.num_items = len(csv)
    
        self.metrics = {col: torch.as_tensor(csv[col]).float() if col != 'PTID' else csv[col] for col in csv.columns} 
        

    def __len__(self):
        return self.num_items
    
    def __getitem__(self, index): 
        item = {col: values[index] for col, values in self.metrics.items()}

        img_path = os.path.join(self.base_path, f'slice_{int(item["slice_number"])}_bias_corrected_registered_ADNI_{item["PTID"]}.png')
        img = torch.tensor(imread(img_path, as_gray=True), dtype=torch.float32)
        
        # You can also load the images directly using PIL but then you can't clip borders as easily
        # img = Image.open(img_path).convert('L')

        # clip image border from PNG files 
        border_clip = 8 
        img = img[border_clip:-border_clip, border_clip:-border_clip]
        
        transform_list = []
        transform_list += [tv.transforms.ToPILImage()]
        if self.crop_type is not None:
            if self.crop_type == 'center':
                transform_list += [tv.transforms.CenterCrop(self.crop_size)]
            elif self.crop_type == 'random':
                transform_list += [tv.transforms.RandomCrop(self.crop_size)]
            else:
                raise ValueError('unknown crop type: {}'.format(self.crop_type))

        if self.downsample is not None and self.downsample > 1:
            transform_list += [tv.transforms.Resize(tuple(np.array(self.crop_size) // self.downsample))]

        transform_list += [tv.transforms.ToTensor()]

        img = tv.transforms.Compose(transform_list)(img)

        item['x'] = img # the 'x' here used to be 'image'

        return item 