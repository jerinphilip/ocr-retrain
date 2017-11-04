'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class hwnet_baseline(nn.Module):
    def __init__(self, numClasses=10000):
        super(hwnet_baseline,self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        
        self.fc1 = nn.Linear(512*6*16, 2048)
        self.bn6 = nn.BatchNorm1d(2048)
        
        self.fc2 = nn.Linear(2048, 2048)
        self.bn7 = nn.BatchNorm1d(2048)
        
        self.fc3 = nn.Linear(2048,numClasses)
        
        #Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def resetLastLayer(self, num_classes):
        self.fc3 = nn.Linear(2048,num_classes)
        n = self.fc3.in_features * self.fc3.out_features
        self.fc3.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x, roi):
        out = F.pad(x,(2,2,2,2),mode='replicate')
        out = self.pool1(F.relu(self.bn1(self.conv1(out))))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.pool4(F.relu(self.bn4(self.conv4(out))))
        out = F.relu(self.bn5(self.conv5(out)))

        out = out.view(-1,512*6*16)
        out = F.relu(self.bn6(self.fc1(out)))
        
        outFeat = self.bn7(self.fc2(out))
        out = F.relu(outFeat)
        
        out = self.fc3(out)

        return out, outFeat
