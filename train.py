import torch
import torch.nn as nn
from torchvision import models, transforms, datasets

from freq_to_bayes import MCDropoutVI, MeanFieldVI
#from modules.dropout import MCDropout

import cv2
import numpy as np
import copy

dtype = torch.FloatTensor

img = np.moveaxis(cv2.imread('magnetit.jpg'), -1, 0) / 255

img_torch = torch.from_numpy(img).unsqueeze(0).type(dtype)

net = nn.Sequential()

net.add_module('conv1', nn.Conv2d(3, 16, 3))
net.add_module('conv2', nn.Conv2d(16, 16, 3))
net.add_module('conv3', nn.Conv2d(16, 3, 3))

# vi_net = MCDropoutVI(net, '2d', 0)
# out = vi_net(img_torch)

# out_seq = img_torch
# for m in net.children():
#     out_seq = m(out_seq)

resnet = models.resnet18()
resnet_vi = MeanFieldVI(resnet)
# resnet_vi = MCDropoutVI(resnet, dropout_type='adaptive', dropout_p=0, deterministic_output=False)
print(resnet_vi)
print(img_torch.size())
out = resnet(img_torch)
out_vi = resnet_vi(img_torch)

optim1 = torch.optim.Adam(resnet_vi.parameters(), lr=0.001, weight_decay=1)
optim2 = torch.optim.Adam(resnet_vi.net.parameters(), lr=0.001, weight_decay=1)
print(optim1.state_dict())
print(optim2.state_dict())
#print(out == out_vi)
#print(resnet_vi)
# resnet2 = copy.deepcopy(resnet)
#
# out = resnet(img_torch)
# for m in resnet.modules():
#     print(m)
    #out_seq = m(out_seq)

# print(out == out_seq)
