# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 14:10:15 2022

@author: johnsonlok
"""

import torch
#import scipy.io 
import scipy.io as sio
#import spectral
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Conv1d(1, 20, kernel_size=48, stride=1, padding=0, bias=True)#ks=24,output=201
#        self.layer2 = torch.nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.layer2 = torch.nn.BatchNorm1d(20)
        self.layer3 = torch.nn.Linear(8020, 100)
        self.layer4 = torch.nn.PReLU(num_parameters=100,init=0.25)
        self.layer6 = torch.nn.PReLU(num_parameters=20,init=0.25)
        self.layer5 = torch.nn.Linear(100, num_classes)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer6(x)
        x=x.reshape(x.size(0),-1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.nn.functional.dropout(x,p = 0.1)
        x = self.layer5(x)
#        x = torch.nn.functional.softmax(x)
        return x

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
'''
images = sio.loadmat('./hyperspectral_datas/salina/data/Salinas_corrected')['salinas_corrected']
labels = sio.loadmat('./hyperspectral_datas/salina/data/Salinas_gt')['salinas_gt']

zero512_217 = np.zeros([512,217])
images = np.insert(images,108, zero512_217, 2)
images = np.insert(images,109, zero512_217, 2)
images = np.insert(images,110, zero512_217, 2)
images = np.insert(images,111, zero512_217, 2)
images = np.insert(images,112, zero512_217, 2)
images = np.insert(images,154, zero512_217, 2)
images = np.insert(images,155, zero512_217, 2)
images = np.insert(images,156, zero512_217, 2)
images = np.insert(images,157, zero512_217, 2)
images = np.insert(images,158, zero512_217, 2)
images = np.insert(images,159, zero512_217, 2)
images = np.insert(images,160, zero512_217, 2)
images = np.insert(images,161, zero512_217, 2)
images = np.insert(images,162, zero512_217, 2)
images = np.insert(images,163, zero512_217, 2)
images = np.insert(images,164, zero512_217, 2)
images = np.insert(images,165, zero512_217, 2)
images = np.insert(images,166, zero512_217, 2)
images = np.insert(images,167, zero512_217, 2)
images = np.insert(images,223, zero512_217, 2)

new_images = np.zeros([512,217,224])
print('starting calculating 3x3 mean, pls ingore the runtime warning.')
for p in range(np.shape(images)[2]):
    for j in range(np.shape(images)[0]):
        for k in range(np.shape(images)[1]):
           if((1<j<500) & (1<k<215)):
                new_images[j][k][p] = (images[j+2][k-2][p] + images[j-2][k-1][p] + images[j-2][k][p]  +  images[j-2][k+1][p] + images[j-2][k+2][p] + 
                                       images[j-1][k-2][p] + images[j-1][k-1][p] + images[j-1][k][p]  + images[j-1][k+1][p]  + images[j-1][k+2][p] +
                                       images[j][k-2][p]   + images[j][k-1][p]   + images[j][k][p]    + images[j][k+1][p]    + images[j][k+2][p]   +
                                       images[j+1][k-2][p] + images[j+1][k-1][p] + images[j+1][k][p]  + images[j+1][k+1][p]  + images[j+1][k+2][p] +
                                       images[j+2][k-2][p] +images[j+2][k-1][p]  + images[j+2][k][p]  + images[j+2][k+1][p]  +images[j+2][k+2][p]) / 25
           else:
                new_images[j][k][p] = images[j][k][p]
    print('calculating mean on page: ',p,'/',np.shape(images)[2]-1)
    
avg_images = new_images

print('starting calculating std, pls wait.')
for p in range((np.shape(images)[2]*2)):#calculate to page 447.
    if(p % 2 != 0):
        print('calculating std on page: ',p,'/',np.shape(images)[2]*2-1)
        new_images = np.insert(new_images,p,zero512_217,axis = 2)
        for j in range(np.shape(images)[0]):
            for k in range(np.shape(images)[1]):
               if((1<j<500) & (1<k<215)):
                   stdarray = (images[j-2][k-2][p//2],images[j-2][k-1][p//2], images[j-2][k][p//2], images[j-2][k+1][p//2],images[j-2][k+2][p//2],
                               images[j-1][k-2][p//2],images[j-1][k-1][p//2], images[j-1][k][p//2], images[j-1][k+1][p//2],images[j-1][k+2][p//2],
                               images[j][k-2][p//2],  images[j][k-1][p//2],   images[j][k][p//2],   images[j][k+1][p//2],  images[j][k+2][p//2],
                               images[j+1][k-2][p//2],images[j+1][k-1][p//2], images[j+1][k][p//2], images[j+1][k+1][p//2],images[j+1][k+2][p//2],
                               images[j+2][k-2][p//2],images[j+2][k-1][p//2], images[j+2][k][p//2], images[j+2][k+1][p//2],images[j+2][k+2][p//2]) #p = 1, p(std) comes from p = 0; p = 3, p(std) comes from p = 1;
                   new_images[j][k][p] = np.std(stdarray)
    else:
        for j in range(np.shape(images)[0]):
            for k in range(np.shape(images)[1]):
                new_images[j][k][p] = avg_images[j][k][p//2]    #p = 0, copy p(ori) = 0; p = 2, copy p(ori) = 1; p = 4, copy p(ori) = 2; p = 6, copy p(ori) = 3

new_images = np.around(new_images)
images = new_images.astype(np.int16)

sio.savemat('./hyperspectral_datas/salina/data/Salinas_corrected5x5std.mat',{'salinas_corrected':images})'''
images = sio.loadmat('./hyperspectral_datas/salina/data/Salinas_corrected5x5std')['salinas_corrected']
labels = sio.loadmat('./hyperspectral_datas/salina/data/Salinas_gt')['salinas_gt']

images = images.reshape((111104,1,448))
labels = labels.flatten()

#part of pre-process, remove class == 0 in data, which means normal grand.
zero = np.where(labels == 0)
no0labels = np.delete(labels,zero)
no0images = np.delete(images,zero,axis=0)
for l in range(no0labels.size):
    no0labels[l] -= 1
num_classes = np.max(no0labels) + 1

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
BGW1_train,BGW1_test,BGW1_trainy,BGW1_testy = train_test_split(no0images[BGW1],no0labels[BGW1],train_size = 200)
BGW2_train,BGW2_test,BGW2_trainy,BGW2_testy = train_test_split(no0images[BGW2],no0labels[BGW2],train_size = 200)
FAL_train,FAL_test,FAL_trainy,FAL_testy = train_test_split(no0images[FAL],no0labels[FAL],train_size = 200)
FRP_train,FRP_test,FRP_trainy,FRP_testy = train_test_split(no0images[FRP],no0labels[FRP],train_size = 200)
FS_train,FS_test,FS_trainy,FS_testy = train_test_split(no0images[FS],no0labels[FS],train_size = 200)
STU_train,STU_test,STU_trainy,STU_testy = train_test_split(no0images[STU],no0labels[STU],train_size = 200)
CELE_train,CELE_test,CELE_trainy,CELE_testy = train_test_split(no0images[CELE],no0labels[CELE],train_size = 200)
GU_train,GU_test,GU_trainy,GU_testy = train_test_split(no0images[GU],no0labels[GU],train_size = 200)
SVD_train,SVD_test,SVD_trainy,SVD_testy = train_test_split(no0images[SVD],no0labels[SVD],train_size = 200)
CSGW_train,CSGW_test,CSGW_trainy,CSGW_testy = train_test_split(no0images[CSGW],no0labels[CSGW],train_size = 200)
L4_train,L4_test,L4_trainy,L4_testy = train_test_split(no0images[L4],no0labels[L4],train_size = 200)
L5_train,L5_test,L5_trainy,L5_testy = train_test_split(no0images[L5],no0labels[L5],train_size = 200)
L6_train,L6_test,L6_trainy,L6_testy = train_test_split(no0images[L6],no0labels[L6],train_size = 200)
L7_train,L7_test,L7_trainy,L7_testy = train_test_split(no0images[L7],no0labels[L7],train_size = 200)
VU_train,VU_test,VU_trainy,VU_testy = train_test_split(no0images[VU],no0labels[VU],train_size = 200)
VVT_train,VVT_test,VVT_trainy,VVT_testy = train_test_split(no0images[VVT],no0labels[VVT],train_size = 200)

x_train = np.concatenate((BGW1_train,BGW2_train,FAL_train,FRP_train,FS_train,STU_train,CELE_train,GU_train,SVD_train,CSGW_train,L4_train,L5_train,L6_train,L7_train,VU_train,VVT_train))
y_train = np.concatenate((BGW1_trainy,BGW2_trainy,FAL_trainy,FRP_trainy,FS_trainy,STU_trainy,CELE_trainy,GU_trainy,SVD_trainy,CSGW_trainy,L4_trainy,L5_trainy,L6_trainy,L7_trainy,VU_trainy,VVT_trainy))
x_test = np.concatenate((BGW1_test,BGW2_test,FAL_test,FRP_test,FS_test,STU_test,CELE_test,GU_test,SVD_test,CSGW_test,L4_test,L5_test,L6_test,L7_test,VU_test,VVT_test))
y_test = np.concatenate((BGW1_testy,BGW2_testy,FAL_testy,FRP_testy,FS_testy,STU_testy,CELE_testy,GU_testy,SVD_testy,CSGW_testy,L4_testy,L5_testy,L6_testy,L7_testy,VU_testy,VVT_testy))

#suit the data format
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
test_dataset0 = TensorDataset(torch.tensor(np.rint(BGW1_test)),torch.tensor(BGW1_testy))
test_dataset1 = TensorDataset(torch.tensor(np.rint(BGW2_test)),torch.tensor(BGW2_testy))
test_dataset2 = TensorDataset(torch.tensor(np.rint(FAL_test)),torch.tensor(FAL_testy))
test_dataset3 = TensorDataset(torch.tensor(np.rint(FRP_test)),torch.tensor(FRP_testy))
test_dataset4 = TensorDataset(torch.tensor(np.rint(FS_test)),torch.tensor(FS_testy))
test_dataset5 = TensorDataset(torch.tensor(np.rint(STU_test)),torch.tensor(STU_testy))
test_dataset6 = TensorDataset(torch.tensor(np.rint(CELE_test)),torch.tensor(CELE_testy))
test_dataset7 = TensorDataset(torch.tensor(np.rint(GU_test)),torch.tensor(GU_testy))
test_dataset8 = TensorDataset(torch.tensor(np.rint(SVD_test)),torch.tensor(SVD_testy))
test_dataset9 = TensorDataset(torch.tensor(np.rint(CSGW_test)),torch.tensor(CSGW_testy))
test_dataset10 = TensorDataset(torch.tensor(np.rint(L4_test)),torch.tensor(L4_testy))
test_dataset11 = TensorDataset(torch.tensor(np.rint(L5_test)),torch.tensor(L5_testy))
test_dataset12 = TensorDataset(torch.tensor(np.rint(L6_test)),torch.tensor(L6_testy))
test_dataset13 = TensorDataset(torch.tensor(np.rint(L7_test)),torch.tensor(L7_testy))
test_dataset14 = TensorDataset(torch.tensor(np.rint(VU_test)),torch.tensor(VU_testy))
test_dataset15 = TensorDataset(torch.tensor(np.rint(VVT_test)),torch.tensor(VVT_testy))

#load data and target
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=10, shuffle=True)    
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=10, shuffle=False)
test_loader0 = torch.utils.data.DataLoader(dataset=test_dataset0,batch_size=10, shuffle=False)
test_loader1 = torch.utils.data.DataLoader(dataset=test_dataset1,batch_size=10, shuffle=False)
test_loader2 = torch.utils.data.DataLoader(dataset=test_dataset2,batch_size=10, shuffle=False)
test_loader3 = torch.utils.data.DataLoader(dataset=test_dataset3,batch_size=10, shuffle=False)
test_loader4 = torch.utils.data.DataLoader(dataset=test_dataset4,batch_size=10, shuffle=False)
test_loader5 = torch.utils.data.DataLoader(dataset=test_dataset5,batch_size=10, shuffle=False)
test_loader6 = torch.utils.data.DataLoader(dataset=test_dataset6,batch_size=10, shuffle=False)
test_loader7 = torch.utils.data.DataLoader(dataset=test_dataset7,batch_size=10, shuffle=False)
test_loader8 = torch.utils.data.DataLoader(dataset=test_dataset8,batch_size=10, shuffle=False)
test_loader9 = torch.utils.data.DataLoader(dataset=test_dataset9,batch_size=10, shuffle=False)
test_loader10 = torch.utils.data.DataLoader(dataset=test_dataset10,batch_size=10, shuffle=False)
test_loader11 = torch.utils.data.DataLoader(dataset=test_dataset11,batch_size=10, shuffle=False)
test_loader12 = torch.utils.data.DataLoader(dataset=test_dataset12,batch_size=10, shuffle=False)
test_loader13 = torch.utils.data.DataLoader(dataset=test_dataset13,batch_size=10, shuffle=False)
test_loader14 = torch.utils.data.DataLoader(dataset=test_dataset14,batch_size=10, shuffle=False)
test_loader15 = torch.utils.data.DataLoader(dataset=test_dataset15,batch_size=10, shuffle=False)

#training parameters
learning_rate = 0.01
num_epochs = 300
model = ConvNet(num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay = 5e-4)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (x_train, y_train) in enumerate(train_loader):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        # Forward pass
        outputs = model(x_train)
        #loss = criterion(outputs, y_train)
        loss = criterion(outputs, y_train)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 60 == 0:
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
    
    correct = 0
    total = 0
    for BGW1_test, BGW1_testy in test_loader0:
        outputs = model(BGW1_test)
        _, predicted = torch.max(outputs.data, 1)
        total += BGW1_testy.size(0)
        correct += (predicted == BGW1_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for BGW2_test, BGW2_testy in test_loader1:
        outputs = model(BGW1_test)
        _, predicted = torch.max(outputs.data, 1)
        total += BGW1_testy.size(0)
        correct += (predicted == BGW1_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for FAL_test, FAL_testy in test_loader2:
        outputs = model(FAL_test)
        _, predicted = torch.max(outputs.data, 1)
        total += FAL_testy.size(0)
        correct += (predicted == FAL_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for FRP_test, FRP_testy in test_loader3:
        outputs = model(FRP_test)
        _, predicted = torch.max(outputs.data, 1)
        total += FRP_testy.size(0)
        correct += (predicted == FRP_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for FS_test, FS_testy in test_loader4:
        outputs = model(FS_test)
        _, predicted = torch.max(outputs.data, 1)
        total += FS_testy.size(0)
        correct += (predicted == FS_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for STU_test, STU_testy in test_loader5:
        outputs = model(STU_test)
        _, predicted = torch.max(outputs.data, 1)
        total += STU_testy.size(0)
        correct += (predicted == STU_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for CELE_test, CELE_testy in test_loader6:
        outputs = model(CELE_test)
        _, predicted = torch.max(outputs.data, 1)
        total += CELE_testy.size(0)
        correct += (predicted == CELE_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for GU_test, GU_testy in test_loader7:
        outputs = model(GU_test)
        _, predicted = torch.max(outputs.data, 1)
        total += GU_testy.size(0)
        correct += (predicted == GU_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for SVD_test, SVD_testy in test_loader8:
        outputs = model(SVD_test)
        _, predicted = torch.max(outputs.data, 1)
        total += SVD_testy.size(0)
        correct += (predicted == SVD_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for CSGW_test, CSGW_testy in test_loader9:
        outputs = model(CSGW_test)
        _, predicted = torch.max(outputs.data, 1)
        total += CSGW_testy.size(0)
        correct += (predicted == CSGW_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for L4_test, L4_testy in test_loader10:
        outputs = model(L4_test)
        _, predicted = torch.max(outputs.data, 1)
        total += L4_testy.size(0)
        correct += (predicted == L4_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for L5_test, L5_testy in test_loader11:
        outputs = model(L5_test)
        _, predicted = torch.max(outputs.data, 1)
        total += L5_testy.size(0)
        correct += (predicted == L5_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for L6_test, L6_testy in test_loader12:
        outputs = model(L6_test)
        _, predicted = torch.max(outputs.data, 1)
        total += L6_testy.size(0)
        correct += (predicted == L6_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for L7_test, L7_testy in test_loader13:
        outputs = model(L7_test)
        _, predicted = torch.max(outputs.data, 1)
        total += L7_testy.size(0)
        correct += (predicted == L7_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for VU_test, VU_testy in test_loader14:
        outputs = model(VU_test)
        _, predicted = torch.max(outputs.data, 1)
        total += VU_testy.size(0)
        correct += (predicted == VU_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
    correct = 0
    total = 0
    for VVT_test, VVT_testy in test_loader15:
        outputs = model(VVT_test)
        _, predicted = torch.max(outputs.data, 1)
        total += VVT_testy.size(0)
        correct += (predicted == VVT_testy).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
    
torch.save(model, 'svmodel.pkl')
torch.save(model.state_dict(), 'svparameter.pkl')