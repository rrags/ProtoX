import numpy as np
import torch as th
import torch.nn as nn
import torchvision.models as models

import torch_optimizer
from torch.autograd import Variable

import torch.nn.functional as F
import os
import time
import random
import sys
import torch

from matplotlib import pyplot as plt
import math
import gc

import pickle


import wandb
from skimage.transform import resize

import torch.nn as nn
import torch.nn.functional as F

#https://discuss.pytorch.org/t/positive-weights/19701/7
class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())


class ProtoIsoResNet(nn.Module):
    def __init__(self,
                 prototype_shape=(60,512*3*3),
                 num_actions=6,
                 init_weights=True,
                 beta=.05,
                 tl=True,
                 frame_stack=4,
                 sim_method=0
    ):
        super(ProtoIsoResNet, self).__init__()
        
        self.num_actions = num_actions
        self.num_prototypes = prototype_shape[0]
        self.prototype_shape = prototype_shape
        self.beta = beta
        self.sim_method = sim_method
                
        print('THIS PROTOPNET IS USING SIM SCORES FOR LOGITS')
        
        if not tl:
            resnet = models.resnet18(pretrained=False)
            resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7,stride=2,padding=3,bias=False)
        else:
            resnet = models.resnet18(pretrained=False)
            resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7,stride=2,padding=3,bias=False)
            print('TL resnet encoder from deepcluster!')
            init_weights = False
            resnet_tl_state = torch.load('enc40.pt').state_dict()
            resnet.load_state_dict(resnet_tl_state,strict=False)
        

        modules = list(resnet.children())[:-2]
        
  
        self.convunit = nn.Sequential(*modules)

        for param in self.convunit.parameters():
            param.requires_grad = False
            
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)


        self.isometry = nn.Linear(512*3*3, prototype_shape[1], bias = False)
        if prototype_shape[1] == 512*3*3:
            self.isometry.weight.data.copy_(torch.eye(512*3*3))
        
        self.last_layer = nn.Linear(self.num_prototypes, self.num_actions, bias=False)
        #self.last_layer = PositiveLinear(self.num_prototypes, self.num_actions)

        assert(self.num_prototypes % self.num_actions == 0)
        
        #one-hot matrix for prototype action label
        self.prototype_action_identity = torch.zeros(self.num_prototypes, self.num_actions)
        num_prototypes_per_action = self.num_prototypes // self.num_actions
        for j in range(self.num_prototypes):
            self.prototype_action_identity[j, j // num_prototypes_per_action] = 1

        if init_weights:
            print('Doing weight init')
            self._init_weights()
    
    def conv_features(self, x):
        return self.convunit(x)
    
    def prototype_distances(self, x):
        conv_output = self.conv_features(x)
        conv_flat = conv_output.flatten(1)
        iso_trans = self.isometry(conv_flat)

        return torch.cdist(iso_trans, self.prototype_vectors[None]).squeeze(0)

        
    def forward(self, x):
        distances = self.prototype_distances(x)
        if self.sim_method == 0:
            sim_scores = torch.exp(-1*self.beta*torch.abs(distances))
        else:
            #eps too small
            sim_scores = torch.log((distances + 1)/(distances + 1e-10))
            
        logits = self.last_layer(sim_scores)
        #logits = self.last_layer(distances)
        return logits, distances
        #return F.softmax(logits), min_distances
    
    def push_forward(self, x):
        conv_output = self.conv_features(x)
        conv_flat = conv_output.flatten(1)
        iso_trans = self.isometry(conv_flat)
        
        dists = torch.cdist(iso_trans, self.prototype_vectors[None]).squeeze(0)
                
        return iso_trans, dists
    
    def select_action(self, x):
        action_prob, min_dists = self.forward(x)
        action = action_prob.multinomial(1)
        return action, min_dists
    
    def targeting_prob(self, x, labels):
        action_prob, _ = self.forward(x)
        return action_prob.gather(1, labels)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_action_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations 
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _init_weights(self):
        for m in self.convunit:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='selu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print('Setting incorrect weight strength due to positive linear!')
        self.set_last_layer_incorrect_connection(incorrect_strength=-.5)

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)
        self.kl = 0
        
        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        
        z = mu + torch.exp(logvar) * self.N.sample(mu.shape)
        self.kl = (torch.exp(logvar)**2 + mu**2 - logvar - 1/2).sum()
        return z#mu, logvar

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        #print('decviewsize: ', x.shape)
        x = F.interpolate(x, scale_factor=4)
        #print('interp size: ', x.shape)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=1.3125)
        x = x.view(x.size(0), 4, 84, 84)
        return x

class VAE(nn.Module):

    def __init__(self, z_dim,nc):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim,nc=nc)
        self.decoder = ResNet18Dec(z_dim=z_dim,nc=nc)

    def forward(self, x):
        #mean, logvar = self.encoder(x)
        #z = self.reparameterize(mean, logvar)
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

