import torch
import torch.nn as nn
import numpy as np
import math


# modify the edge detector kernel in such a way that
# it calculates the derivatives in x and y direction
edge_detector_kernel = torch.zeros(2, 1, 2, 2)
edge_detector_kernel[0,0,0,0] = -1
edge_detector_kernel[0,0,1,0] =  1
edge_detector_kernel[1,0,0,0] = -1
edge_detector_kernel[1,0,0,1] =  1

#edge_detector_kernel[1,0] = torch.tensor([[1,0],[-1,0]])

 

class Conv2d(nn.Module):
    
    def __init__(self, kernel, padding=0, stride=1):
        super().__init__()
        self.kernel = nn.Parameter(kernel)
        self.padding = ZeroPad2d(padding)
        self.stride = stride
        
    def forward(self, x):
        before_padding = x
        x = self.padding(x)
        # Implement the convolution of x with self.kernel
        # using self.stride as stride
        
        #output = torch.zeros(self.kernel.shape[0], math.floor((x.shape[1] + 2 - self.kernel.shape[2])/self.stride)+1, math.floor((x.shape[2] + 2 - self.kernel.shape[3])/self.stride)+1)
        
        output = torch.zeros(self.kernel.shape[0], math.floor((before_padding.shape[1] + (x.shape[1] - before_padding.shape[1])  - self.kernel.shape[2])/self.stride)+1, math.floor((before_padding.shape[2] + (x.shape[2] - before_padding.shape[2]) - self.kernel.shape[3])/self.stride)+1)
        
        for out_ch in range(output.shape[0]):
            start_row = 0
            stop_row = 0
            for row in range(output.shape[1]):
                start_col = 0
                stop_col = 0
                for col in range(output.shape[2]):
                    for channel in range(self.kernel.shape[1]):
                        output[out_ch,row,col] = output[out_ch,row,col] + torch.sum(x[channel, start_row:(self.kernel.shape[2] + stop_row), start_col:(self.kernel.shape[3] + stop_col)] * self.kernel[out_ch, channel])
                    start_col = start_col + self.stride
                    stop_col = stop_col + self.stride
                start_row = start_row + self.stride
                stop_row = stop_row + self.stride
                    
        '''output[0,0,0] = torch.sum(x[0, 0:4, 0:5] * self.kernel[0,0]) + torch.sum(x[1, 0:4, 0:5] * self.kernel[0,1]) + torch.sum(x[2, 0:4, 0:5] * self.kernel[0,2])
        output[0,0,1] = torch.sum(x[0, 0:4, 3:8] * self.kernel[0,0]) + torch.sum(x[1, 0:4, 3:8] * self.kernel[0,1]) + torch.sum(x[2, 0:4, 3:8] * self.kernel[0,2])
        output[0,1,0] = torch.sum(x[0, 3:7, 0:5] * self.kernel[0,0]) + torch.sum(x[1, 3:7, 0:5] * self.kernel[0,1]) + torch.sum(x[2, 3:7, 0:5] * self.kernel[0,2])
        output[0,1,1] = torch.sum(x[0, 3:7, 3:8] * self.kernel[0,0]) + torch.sum(x[1, 3:7, 3:8] * self.kernel[0,1]) + torch.sum(x[2, 3:7, 3:8] * self.kernel[0,2])
        
        output[1,0,0] = torch.sum(x[0, 0:4, 0:5] * self.kernel[1,0]) + torch.sum(x[1, 0:4, 0:5] * self.kernel[1,1]) + torch.sum(x[2, 0:4, 0:5] * self.kernel[1,2])
        output[1,0,1] = torch.sum(x[0, 0:4, 3:8] * self.kernel[1,0]) + torch.sum(x[1, 0:4, 3:8] * self.kernel[1,1]) + torch.sum(x[2, 0:4, 3:8] * self.kernel[1,2])
        output[1,1,0] = torch.sum(x[0, 3:7, 0:5] * self.kernel[1,0]) + torch.sum(x[1, 3:7, 0:5] * self.kernel[1,1]) + torch.sum(x[2, 3:7, 0:5] * self.kernel[1,2])
        output[1,1,1] = torch.sum(x[0, 3:7, 3:8] * self.kernel[1,0]) + torch.sum(x[1, 3:7, 3:8] * self.kernel[1,1]) + torch.sum(x[2, 3:7, 3:8] * self.kernel[1,2])'''
        return output
        #pass


class ZeroPad2d(nn.Module):
    
    def __init__(self, padding):
        super().__init__()
        self.padding = padding
        
    def forward(self, x):
        # for input of shape B x C x H x W
        # return tensor zero padded equally at left, right,
        # top, bottom such that the output is of size
        # B x C x (H + 2 * self.padding) x (W + 2 * self.padding)
        if(self.padding > 0):
            #dum_tensor = torch.zeros((x.shape[1] + (2 * padding)),(x.shape[2] + (2 * padding)))
            dum = torch.zeros((x.shape[0]),(x.shape[1] + (2 * self.padding)),(x.shape[2] + (2 * self.padding)))
        
            #for batch in range(zero_dum.shape[0]):

            for channel in range(x.shape[0]):      # 3
                for row in range(x.shape[1]):      # 6
                     for col in range(x.shape[2]): # 8
                        dum[channel,row+1,col+1] = x[channel,row,col]  
                        
            return dum
        else:
            return x
