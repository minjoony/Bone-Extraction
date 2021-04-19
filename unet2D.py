import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def conv(in_dim, out_dim, kernel_size):
            stride = 1
            padding = 1

            model = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.BatchNorm2d(num_features = out_dim),
                nn.ReLU()
            )
            return model

        def deconv(in_dim, out_dim, kernel_size):
            stride = 2
            padding = 1

            model = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
            return model
        
        # Contracting path (Encoder)
        num_filter = 32
        
        self.conv1_1 = conv(in_dim=1, out_dim=num_filter, kernel_size=3)
        self.conv1_2 = conv(num_filter, num_filter, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = conv(num_filter, num_filter*2, 3)
        self.conv2_2 = conv(num_filter*2, num_filter*2, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = conv(num_filter*2, num_filter*4, 3)
        self.conv3_2 = conv(num_filter*4, num_filter*4, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = conv(num_filter*4, num_filter*8, 3)
        self.conv4_2 = conv(num_filter*8, num_filter*8, 3)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = conv(num_filter*8, num_filter*16, 3)
        self.conv5_2 = conv(num_filter*16, num_filter*16, 3)
        
        # Expansive path (Decoder)
        self.deconv6 = deconv(in_dim = num_filter*16, out_dim = num_filter*8, kernel_size = 4)
        self.conv6_1 = conv(num_filter*16, num_filter*8, 3)
        self.conv6_2 = conv(num_filter*8, num_filter*8, 3)
        
        self.deconv7 = deconv(num_filter*8, num_filter*4, 4)
        self.conv7_1 = conv(num_filter*8, num_filter*4, 3)
        self.conv7_2 = conv(num_filter*4, num_filter*4, 3)
        
        self.deconv8 = deconv(num_filter*4, num_filter*2, 4)
        self.conv8_1 = conv(num_filter*4, num_filter*2, 3)
        self.conv8_2 = conv(num_filter*2, num_filter*2, 3)
        
        self.deconv9 = deconv(num_filter*2, num_filter, 4)
        self.conv9_1 = conv(num_filter*2, num_filter, 3)
        self.conv9_2 = conv(num_filter, num_filter, 3)
        
        self.out = nn.Conv2d(in_channels=num_filter, out_channels=1, kernel_size=1, stride=1, padding=0)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
#                 nn.init.normal_(m.weight.data, mean=0, std=0.01)
                nn.init.xavier_normal_(m.weight.data)
#                 nn.init.kaiming_normal_(m.weight.data)
    
    def forward(self, input, prt=False):
        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)
        
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)
        
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.pool4(conv4_2)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)

        deconv6 = self.deconv6(conv5_2)
        concat6 = torch.cat((deconv6, conv4_2), dim=1)
        conv6_1 = self.conv6_1(concat6)
        conv6_2 = self.conv6_2(conv6_1)
        
        deconv7 = self.deconv7(conv6_2)
        concat7 = torch.cat((deconv7, conv3_2), dim=1)
        conv7_1 = self.conv7_1(concat7)
        conv7_2 = self.conv7_2(conv7_1)
        
        deconv8 = self.deconv8(conv7_2)
        concat8 = torch.cat((deconv8, conv2_2), dim=1)
        conv8_1 = self.conv8_1(concat8)
        conv8_2 = self.conv8_2(conv8_1)
        
        deconv9 = self.deconv9(conv8_2)
        concat9 = torch.cat((deconv9, conv1_2), dim=1)
        conv9_1 = self.conv9_1(concat9)
        conv9_2 = self.conv9_2(conv9_1)
        
        output = self.out(conv9_2)
        
        if(prt == True):
            print("input shape :", input.shape)
            print("")
            
            print("conv1_1 shape :", conv1_1.shape)
            print("conv1_2 shape :", conv1_2.shape)
            print("pool1 shape :", pool1.shape)
            print("")
            
            print("conv2_1 shape :", conv2_1.shape)
            print("conv2_2 shape :", conv2_2.shape)
            print("pool2 shape :", pool2.shape)
            print("")
            
            print("conv3_1 shape :", conv3_1.shape)
            print("conv3_2 shape :", conv3_2.shape)
            print("pool3 shape :", pool3.shape)
            print("")

            print("conv4_1 shape :", conv4_1.shape)
            print("conv4_2 shape :", conv4_2.shape)
            print("pool4 shape :", pool4.shape)
            print("")
            
            print("conv5_1 shape :", conv5_1.shape)
            print("conv5_2 shape :", conv5_2.shape)
            print("")
            
            print("deconv6_shape :", deconv6.shape)
            print("concat6_shape :", concat6.shape)
            print("conv6_1 shape :", conv6_1.shape)
            print("conv6_2 shape :", conv6_2.shape)
            print("")
            
            print("deconv7_shape :", deconv7.shape)
            print("concat7_shape :", concat7.shape)
            print("conv7_1 shape :", conv7_1.shape)
            print("conv7_2 shape :", conv7_2.shape)
            print("")
            
            print("deconv8_shape :", deconv8.shape)
            print("concat8_shape :", concat8.shape)
            print("conv8_1 shape :", conv8_1.shape)
            print("conv8_2 shape :", conv8_2.shape)
            print("")
            
            print("deconv9_shape :", deconv9.shape)
            print("concat9_shape :", concat9.shape)
            print("conv9_1 shape :", conv9_1.shape)
            print("conv9_2 shape :", conv9_2.shape)
            print("")
            
            print("\noutput shape :", output.shape)
        
        return output
