from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import argparse
import cv2
import os
import pdb

from models import *
import synthTransformer as synthTrans
import hwroidataset as hwROIDat

parser = argparse.ArgumentParser(description='PyTorch HWNet Training')

parser.add_argument('--img_folder', help='image root folder', required=True)
parser.add_argument('--test_vocab_file', help='test IAM file', required=True)
parser.add_argument('--savedir',  help='output directory to save files', required=True)
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--pretrained_file',  help='pre trained file path', required=True)
parser.add_argument('--arch', default='resnetROI34', help='architecture selection')

args = parser.parse_args()
print(args)
exit()

if(not os.path.exists(save_dir)):
    os.makedirs(save_dir)

use_cuda = torch.cuda.is_available()

#Transformer
transform_test = transforms.Compose([
    synthTrans.Normalize(),
    synthTrans.ToTensor()
])

#Dataset
testset = hwROIDat.HWRoiDataset(ann_file=args.test_vocab_file,
                                    img_folder=args.img_folder,
                                    randFlag=False,
                                    valFlag = True,
                                    transform=transform_test)
#Dataloader
testloader = DataLoader(testset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

pretrained_path = args.pretrained_path
checkpoint = torch.load(pretrained_path)

net = checkpoint['net']
net.eval()

#Arch Selection
if args.arch=='resnet18' or args.arch=='resnet34':
    featMat = np.zeros((len(testset),512))
elif args.arch=='hwnet' or args.arch=='hwnetDeep' or args.arch=='resnetROI18' or args.arch=='resnetROI34':
    featMat = np.zeros((len(testset),2048))
elif args.arch=='densenet121':
    featMat = np.zeros((len(testset),1024))

fCntr=0
for batch_idx, data in enumerate(testloader):
    print('batch idx: %d'%batch_idx)
    inputs, targets, roi = data['image'], data['label'], data['roi']
    roi[:,0] = torch.arange(0,roi.size()[0])

    if use_cuda:
        inputs, targets, roi = inputs.cuda(), targets.cuda(), roi.cuda()
    inputs = inputs.unsqueeze(1)
    targets = targets.squeeze()

    inputs, targets, roi = Variable(inputs, volatile=True), Variable(targets), Variable(roi)
    
    outputs, outFeats = net(inputs, roi)
    featData = outFeats.cpu().data.numpy()
    
    #Normalize
    normVal = np.sqrt(np.sum(featData**2,axis=1))
    featData = featData/normVal.reshape((targets.size(0),1))
    featMat[fCntr:fCntr+targets.size(0),:] = featData
    fCntr+=targets.size(0)

# Save features
print('Saving features file')
np.save(save_dir+'feats.npy', featMat)
print('Features file saved')
