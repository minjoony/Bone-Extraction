import os
import torch
import cv2
import numpy as np
from PIL import Image


""" Making the dataset """
'''
train_set : [1, 512, 512] x 356 (178 for 'input', 178 for 'mask')
valid_set : [1, 512, 512] x 20 (10 for 'input', 10 for 'mask')
test_set : [1, 512, 512] x 10 (5 for 'input', 5 for 'mask')
'''
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, mask_path, trans=None):
        self.input = self.img_to_numpy(input_path)
        self.mask = self.img_to_numpy(mask_path)
        self.trans = trans
        
    def __len__(self):
        return len(self.input)
        
    def __getitem__(self, idx):
        return self.trans(self.input[idx]), self.trans(self.mask[idx])
    
    def img_to_numpy(self, path):
        '''
        Converting images to list of numpy.
        '''
        file_list = []
        np_list = []

        ### Removing the ".ipynb_checkpoints/" directory to prevent from insertion into Dataset
        if os.path.isdir(path + ".ipynb_checkpoints"):
            shutil.rmtree(path + ".ipynb_checkpoints")
            
        for _file in os.listdir(path):
            file_list.append(os.path.join(path, _file))
        file_list.sort()
        
        ### Saving the dataset as numpy
        for _file in file_list:
            _img = Image.open(_file).convert('L')
            _np = np.asarray(_img)
            _np = cv2.resize(_np, (512, 512), interpolation = cv2.INTER_AREA)

            assert _np.ndim < 3, 'Dim error : There are RGB images in dataset. Convert it to gray scale.'
            assert _np.shape == (512, 512), 'Shape error : Convert it to [512, 512]'

            np_list.append(_np)

        return np_list
