import os
import time
import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from customDataset import CustomDataset
from unet2D import UNet
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from torchvision.utils import save_image

RESULT_PATH = './Results/'
TEST_PATH = './Data/test/'
REAL_PATH = './Data/Real/input/'
SAVE_PATH = './Checkpoints/'

TEST_INPUT_PATH = TEST_PATH + 'input/'
TEST_MASK_PATH = TEST_PATH + 'mask/'

image_size = 512
batch_size = 1


test_set = CustomDataset(input_path = REAL_PATH,
                         mask_path = REAL_PATH,
                         trans = transforms.Compose([transforms.ToTensor()]))

test_loader = DataLoader(test_set,
                          batch_size = batch_size,
                          shuffle = False,
                          num_workers = 4)

model = torch.load(SAVE_PATH + 'Unet.pt')
model.load_state_dict(torch.load(SAVE_PATH + 'Unet_weight_300.pth.tar')['state_dict'])

model.eval()
epoch_loss = 0
i = 0

for test_data in test_loader:
    X_test = test_data[0].cuda()
    Y_test = test_data[1].cuda()
    print(X_test.shape)

    PredX_test = model(X_test.float())
    
    ### Plotting the output image of the network
    save_image(X_test, RESULT_PATH + 'input_' + str(i) + '.png')
#     save_image(Y_test, RESULT_PATH + 'mask' + str(i) + '.png')
    save_image(PredX_test, RESULT_PATH + 'output_' + str(i) + '.png')

    first = Image.open(RESULT_PATH + 'input_' + str(i) + '.png')
#     second = Image.open(RESULT_PATH + 'mask' + str(i) + '.png')
    third = Image.open(RESULT_PATH + 'output_' + str(i) + '.png')

#     new_image = Image.new('RGB', (3*image_size, image_size))
    new_image = Image.new('RGB', (2*image_size, image_size))

    new_image.paste(im=first, box=(0,0))
    new_image.paste(im=third, box=(image_size, 0))

#     new_image.paste(im=second, box=(image_size, 0))
#     new_image.paste(im=third, box=(image_size*2, 0))
    new_image.save(RESULT_PATH + "merged_image_" + str(i) + ".png", "PNG")
    i = i+1
print("Done !!! ")
