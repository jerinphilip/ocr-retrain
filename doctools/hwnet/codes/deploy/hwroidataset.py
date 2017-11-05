from torch.utils.data import Dataset
from torchvision import transforms, utils
from nltk.stem.porter import *
import os

import cv2

import numpy as np
import random
import pdb

class HWRoiDataset(Dataset):
    """IAM dataset"""

    def __init__(self, ann_file, img_folder, randFlag=False, valFlag = False, transform=None, portStem=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.words = []
        self.vocab = {}
        self.vocabIdx = {}
        self.imgPaths = []
        self.imgFolder = img_folder
        
        self.randFlag = randFlag
        self.portStem = portStem

        stemmer = PorterStemmer()

        vCntr=0
        with open(ann_file) as vFile:
            for line in vFile:               
                tempStr = line.split()
                if(valFlag and int(tempStr[3])==1):
                    self.imgPaths.append(tempStr[0])
                elif((not valFlag) and int(tempStr[3])==0):
                    self.imgPaths.append(tempStr[0])

                if tempStr[1] not in self.vocab:
                    self.vocab[tempStr[1]]= vCntr
                    #self.vocabIdx[vCntr] = tempStr[1]
                    if self.portStem:
                        self.vocabIdx[vCntr] = stemmer.stem(tempStr[1])
                    else:
                        self.vocabIdx[vCntr] = tempStr[1]

                    
                    if(valFlag and int(tempStr[3])==1):
                        self.words.append(vCntr)
                    elif((not valFlag) and int(tempStr[3])==0):
                        self.words.append(vCntr)                    
                    vCntr+=1
                else:
                    if(valFlag and int(tempStr[3])==1):
                        self.words.append(self.vocab[tempStr[1]])
                    elif((not valFlag) and int(tempStr[3])==0):
                        self.words.append(self.vocab[tempStr[1]])

        self.transform = transform

    def __len__(self):
        return len(self.words)
    def __getitem__(self, idx):
        
        if self.randFlag:
            fontsize = np.random.randint(32,64)
        else:
            fontsize = 48
        #pdb.set_trace()
        try:
            #Read image
            path = os.path.join(self.imgFolder, self.imgPaths[idx])
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            h,w = image.shape
        except:
            print('Warning: while rendering --%s-- at index %d. Rendering default value' % (self.vocabIdx[self.words[idx]],idx))
            idx=0
            path = os.path.join(self.imgFolder, self.imgPaths[idx])
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            h,w = image.shape
        #pdb.set_trace()
        # if self.randFlag:
        newWidth = np.min((int(np.ceil(((w*1.0)/h) * fontsize)),384-1))
        image = cv2.resize(image,(newWidth,fontsize))
        newImage = np.ones((128,384),dtype=np.float32) * 255.0

        rX = np.random.randint(0,384-newWidth)
        rY = np.random.randint(0,128-64)
        newImage[rY:fontsize+rY, rX:newWidth+rX] = image
        cords = np.where(newImage!=255)
        if cords[0].size==0:
            roi = np.asarray([0.0, 0.0,0.0, 100.0, 100.0],dtype=np.float32)
        else:
            roi = np.asarray([0.0, min(cords[1]),min(cords[0]),max(cords[1]),max(cords[0])],dtype=np.float32)

        # else:
        #     #Just resizing the image to constant height (batch size should be 1 to support variable size inputs)
        #     #ToDO: Correct it
        #     #rX = 0
        #     #rY = 0
        #     newWidth = int(np.ceil(((w*1.0)/h) * fontsize))
        #     tempImage = cv2.resize(image,(newWidth,fontsize))
        #     if(newWidth<96):
        #         newImage = np.ones((fontsize,96),dtype=np.float32) * 255.0
        #         newImage[:,:newWidth] = tempImage
        #     else:
        #         newImage = np.asarray(tempImage,dtype=np.float32)

        #     roi = np.asarray([0.0, 0.0, 0.0, newWidth, fontsize],dtype=np.float32)
        
        label = np.zeros(1,dtype=np.int64)
        label[0] = self.words[idx]

        sample = {'image': newImage, 'gt': self.vocabIdx[self.words[idx]], 'label': label, 'roi': roi}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def getGTtext(self, idx):
        return self.vocabIdx[idx]

    def getVocabSize(self):
        return len(self.vocab)
