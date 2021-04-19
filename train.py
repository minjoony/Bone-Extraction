#
# Description:
#   Network for extraction bone-mask image from DRR image
#

import os
import time
import numpy as np
import torch
import torchvision
import shutil
import matplotlib.pyplot as plt
from customDataset import CustomDataset
from unet2D import UNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from torchvision.utils import save_image


""" File Path """
TRAIN_PATH = './Data/Aug/'
TRAIN_INPUT_PATH = TRAIN_PATH + 'input/'
TRAIN_MASK_PATH = TRAIN_PATH + 'mask/'

VALID_PATH = './Data/valid/'
VALID_INPUT_PATH = VALID_PATH + 'input/'
VALID_MASK_PATH = VALID_PATH + 'mask/'

RESULT_PATH = './Results/'
SAVE_PATH = './Checkpoints/'


""" Constant Variables """
'''
image_size : size of the image (n x n)
learning_rate : learning rate for training
train_epoch : The number of training epochs
batch_size : batch size for training
save_step : Step for saving network
'''
image_size = 512
learning_rate = 1e-3
train_epoch = 300
batch_size = 1
save_step = 10
        

""" Training """
train_set = CustomDataset(input_path = TRAIN_INPUT_PATH,
                          mask_path = TRAIN_MASK_PATH,
                          trans = transforms.Compose([transforms.ToTensor()]))

valid_set = CustomDataset(input_path = VALID_INPUT_PATH,
                          mask_path = VALID_MASK_PATH,
                          trans = transforms.Compose([transforms.ToTensor()]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\ndevice :', device)

### Loading the dataset
train_loader = DataLoader(train_set,
                          batch_size = batch_size,
                          shuffle = False,
                          num_workers = 4)

valid_loader = DataLoader(valid_set,
                          batch_size = batch_size,
                          shuffle = False,
                          num_workers = 4)

### Making the "loss.txt" file to record the loss
txt = open(RESULT_PATH + 'loss.txt', 'w')
txt.write("****************************************\n Loss of U-Net using L1 loss \n****************************************\n")

### Training the network
net = UNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
loss_model = torch.nn.MSELoss()

loss_arr = []
loss_val_arr = []

print("------ Training is started ------")
for epoch in range(train_epoch):
    ### Forward propagation
    net.train()
    epoch_loss = 0

    for train_data in train_loader:
        '''
        X : DRR image ([1, 1, 512, 512])
        Y : Bone-mask image ([1, 1, 512, 512])
        PredX : Prediction of the neural network ([1, 1, 512, 512])
        '''
        X = train_data[0].cuda()
        Y = train_data[1].cuda()
        
        PredX = net(X.float())

        ### Back propagation
        optimizer.zero_grad()
        loss = loss_model(PredX, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    print("Train: EPOCH %04d / %04d | LOSS %.4f" %(epoch+1, train_epoch, epoch_loss/len(train_loader)))
    txt.write("Train: EPOCH %04d / %04d | LOSS %.4f\n" %(epoch+1, train_epoch, epoch_loss/len(train_loader)))
    loss_arr.append(epoch_loss / len(train_set))
    
    ### Validating the network
    net.eval()
    epoch_loss = 0
    
    for valid_data in valid_loader:
        X_val = valid_data[0].cuda()
        Y_val = valid_data[1].cuda()
        
        PredX_val = net(X_val.float())
        
        loss = loss_model(PredX_val, Y_val)
        epoch_loss += loss.item()
    
    print("Valid: EPOCH %04d / %04d | LOSS %.4f\n" %(epoch+1, train_epoch, epoch_loss/len(valid_loader)))
    txt.write("Valid: EPOCH %04d / %04d | LOSS %.4f\n\n" %(epoch+1, train_epoch, epoch_loss/len(valid_loader)))
    loss_val_arr.append(epoch_loss / len(valid_loader))
    
    ### Saving the network
    if (epoch+1) % save_step == 0:
        torch.save({'state_dict':net.state_dict()}, SAVE_PATH + 'Unet_weight_' + str(epoch+1) + '.pth.tar')

torch.save(net, SAVE_PATH + 'Unet.pt')
txt.close()
print("------ Training is finished ------")


""" Plotting learning curve """
epoch_list = range(1, train_epoch+1)
plt.plot(epoch_list, np.array(loss_arr), 'y')
plt.plot(epoch_list, np.array(loss_val_arr), 'c')
plt.title('Loss Graph')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend((['train loss', 'valid loss']))
plt.savefig(RESULT_PATH + 'loss graph.png')
