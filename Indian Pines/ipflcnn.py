# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 14:10:15 2022

@author: johnsonlok
"""
import torch
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torchvision

class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Conv1d(1, 20, kernel_size=48, stride=1, padding=0, bias=True)#ks=24,output=201
        self.layer2 = torch.nn.BatchNorm1d(20)
        self.layer3 = torch.nn.Linear(8020, 100)
        self.layer4 = torch.nn.PReLU(num_parameters=100,init=0.25)
        self.layer6 = torch.nn.PReLU(num_parameters=20,init=0.25)
        '''for p in self.parameters():
          p.requires_grad = False'''
        self.layer5 = torch.nn.Linear(100, num_classes)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer6(x)
        x = x.reshape(x.size(0),-1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.nn.functional.dropout(x,p = 0.1)
        x = self.layer5(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
images = sio.loadmat('./hyperspectral_datas/salina/data/ip_corrected5x5std')['salinas_corrected']
labels = sio.loadmat('Indian_pines_gt')['indian_pines_gt']

imagesip = images.reshape((21025,1,448))
labelsip = labels.flatten()

#part of pre-process, remove class == 0 in data, which means normal grand.
zero = np.where(labelsip == 0)
no0labelsip = np.delete(labelsip,zero)
no0imagesip = np.delete(imagesip,zero,axis=0)
for i in range(no0labelsip.size):
    no0labelsip[i] -= 1

no0labels = no0labelsip
no0images = no0imagesip

BGW1 = np.where(no0labels == 0)
BGW2 = np.where(no0labels == 1)
FAL = np.where(no0labels == 2)
FRP = np.where(no0labels == 3)
FS = np.where(no0labels == 4)
STU = np.where(no0labels == 5)
CELE = np.where(no0labels == 6)
GU = np.where(no0labels == 7)
SVD = np.where(no0labels == 8)
CSGW = np.where(no0labels == 9)
L4 = np.where(no0labels == 10)
L5 = np.where(no0labels == 11)
L6 = np.where(no0labels == 12)
L7 = np.where(no0labels == 13)
VU = np.where(no0labels == 14)
VVT = np.where(no0labels == 15)

#train eval data split
#x_train,x_test,y_train,y_test = train_test_split(no0images,no0labels,train_size = 3200,stratify = no0labels)
BGW1_train,BGW1_test,BGW1_trainy,BGW1_testy = train_test_split(no0images[BGW1],no0labels[BGW1],train_size = 0.1)
BGW2_train,BGW2_test,BGW2_trainy,BGW2_testy = train_test_split(no0images[BGW2],no0labels[BGW2],train_size = 0.1)
FAL_train,FAL_test,FAL_trainy,FAL_testy = train_test_split(no0images[FAL],no0labels[FAL],train_size = 0.1)
FRP_train,FRP_test,FRP_trainy,FRP_testy = train_test_split(no0images[FRP],no0labels[FRP],train_size = 0.1)
FS_train,FS_test,FS_trainy,FS_testy = train_test_split(no0images[FS],no0labels[FS],train_size = 0.1)
STU_train,STU_test,STU_trainy,STU_testy = train_test_split(no0images[STU],no0labels[STU],train_size = 0.1)
CELE_train,CELE_test,CELE_trainy,CELE_testy = train_test_split(no0images[CELE],no0labels[CELE],train_size = 0.1)
GU_train,GU_test,GU_trainy,GU_testy = train_test_split(no0images[GU],no0labels[GU],train_size = 0.1)
SVD_train,SVD_test,SVD_trainy,SVD_testy = train_test_split(no0images[SVD],no0labels[SVD],train_size = 0.1)
CSGW_train,CSGW_test,CSGW_trainy,CSGW_testy = train_test_split(no0images[CSGW],no0labels[CSGW],train_size = 0.1)
L4_train,L4_test,L4_trainy,L4_testy = train_test_split(no0images[L4],no0labels[L4],train_size = 0.1)
L5_train,L5_test,L5_trainy,L5_testy = train_test_split(no0images[L5],no0labels[L5],train_size = 0.1)
L6_train,L6_test,L6_trainy,L6_testy = train_test_split(no0images[L6],no0labels[L6],train_size = 0.1)
L7_train,L7_test,L7_trainy,L7_testy = train_test_split(no0images[L7],no0labels[L7],train_size = 0.1)
VU_train,VU_test,VU_trainy,VU_testy = train_test_split(no0images[VU],no0labels[VU],train_size = 0.1)
VVT_train,VVT_test,VVT_trainy,VVT_testy = train_test_split(no0images[VVT],no0labels[VVT],train_size = 0.1)

x_train = np.concatenate((BGW1_train,BGW2_train,FAL_train,FRP_train,FS_train,STU_train,CELE_train,GU_train,SVD_train,CSGW_train,L4_train,L5_train,L6_train,L7_train,VU_train,VVT_train))
y_train = np.concatenate((BGW1_trainy,BGW2_trainy,FAL_trainy,FRP_trainy,FS_trainy,STU_trainy,CELE_trainy,GU_trainy,SVD_trainy,CSGW_trainy,L4_trainy,L5_trainy,L6_trainy,L7_trainy,VU_trainy,VVT_trainy))
x_test = np.concatenate((BGW1_test,BGW2_test,FAL_test,FRP_test,FS_test,STU_test,CELE_test,GU_test,SVD_test,CSGW_test,L4_test,L5_test,L6_test,L7_test,VU_test,VVT_test))
y_test = np.concatenate((BGW1_testy,BGW2_testy,FAL_testy,FRP_testy,FS_testy,STU_testy,CELE_testy,GU_testy,SVD_testy,CSGW_testy,L4_testy,L5_testy,L6_testy,L7_testy,VU_testy,VVT_testy))

x_train = np.rint(x_train)
x_train = torch.tensor(x_train)
x_train = torchvision.transforms.Normalize(mean = 0, std = 1)(x_train)
y_train = torch.tensor(y_train)

x_test = np.rint(x_test)
x_test = torch.tensor(x_test)
x_test = torchvision.transforms.Normalize(mean = 0, std = 1)(x_test)
y_test = torch.tensor(y_test)

#combine data and target
train_dataset = TensorDataset(x_train,y_train)
test_dataset =  TensorDataset(x_test,y_test)

#load data and target
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=10, shuffle=True)    
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=10, shuffle=False)

#training parameters
learning_rate = 0.01
num_epochs = 300
num_classes = 16
model = ConvNet(num_classes).to(device)
model.load_state_dict(torch.load('svparameter.pkl'))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay = 5e-4)
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (x_train, y_train) in enumerate(train_loader):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 102 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        outputs = model(x_test)
        _, predicted = torch.max(outputs.data, 1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
