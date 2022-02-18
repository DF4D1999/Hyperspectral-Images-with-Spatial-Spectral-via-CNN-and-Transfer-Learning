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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

imagesip = sio.loadmat('Indian_pines_corrected')['indian_pines_corrected']
labels = sio.loadmat('Indian_pines_gt')['indian_pines_gt']

zero145_145 = np.zeros([145,145])
imagesip = np.insert(imagesip,104, zero145_145, 2)
imagesip = np.insert(imagesip,105, zero145_145, 2)
imagesip = np.insert(imagesip,106, zero145_145, 2)
imagesip = np.insert(imagesip,107, zero145_145, 2)
imagesip = np.insert(imagesip,108, zero145_145, 2)
imagesip = np.insert(imagesip,150, zero145_145, 2)
imagesip = np.insert(imagesip,151, zero145_145, 2)
imagesip = np.insert(imagesip,152, zero145_145, 2)
imagesip = np.insert(imagesip,153, zero145_145, 2)
imagesip = np.insert(imagesip,154, zero145_145, 2)
imagesip = np.insert(imagesip,155, zero145_145, 2)
imagesip = np.insert(imagesip,156, zero145_145, 2)
imagesip = np.insert(imagesip,157, zero145_145, 2)
imagesip = np.insert(imagesip,158, zero145_145, 2)
imagesip = np.insert(imagesip,159, zero145_145, 2)
imagesip = np.insert(imagesip,160, zero145_145, 2)
imagesip = np.insert(imagesip,161, zero145_145, 2)
imagesip = np.insert(imagesip,162, zero145_145, 2)
imagesip = np.insert(imagesip,163, zero145_145, 2)
imagesip = np.insert(imagesip,219, zero145_145, 2)
imagesip = np.insert(imagesip,220, zero145_145, 2)
imagesip = np.insert(imagesip,221, zero145_145, 2)
imagesip = np.insert(imagesip,222, zero145_145, 2)
imagesip = np.insert(imagesip,223, zero145_145, 2)
images = imagesip

new_images = np.zeros([145,145,224])
print('starting calculating 3x3 mean, pls ingore the runtime warning.')
for p in range(np.shape(images)[2]):
    for j in range(np.shape(images)[0]):
        for k in range(np.shape(images)[1]):
           if((1<j<143) & (1<k<143)):
                new_images[j][k][p] = (images[j+2][k-2][p] + images[j-2][k-1][p] + images[j-2][k][p]  +  images[j-2][k+1][p] + images[j-2][k+2][p] + 
                                       images[j-1][k-2][p] + images[j-1][k-1][p] + images[j-1][k][p]  + images[j-1][k+1][p]  + images[j-1][k+2][p] +
                                       images[j][k-2][p]   + images[j][k-1][p]   + images[j][k][p]    + images[j][k+1][p]    + images[j][k+2][p]   +
                                       images[j+1][k-2][p] + images[j+1][k-1][p] + images[j+1][k][p]  + images[j+1][k+1][p]  + images[j+1][k+2][p] +
                                       images[j+2][k-2][p] +images[j+2][k-1][p]  + images[j+2][k][p]  + images[j+2][k+1][p]  + images[j+2][k+2][p]) / 25
           else:
                new_images[j][k][p] = images[j][k][p]
                
    print('calculating mean on page: ',p,'/',np.shape(imagesip)[2]-1)
    
avg_images = new_images

print('starting calculating std, pls wait.')
for p in range((np.shape(images)[2]*2)):#calculate to page 447.
    if(p % 2 != 0):
        print('calculating std on page: ',p,'/',np.shape(images)[2]*2-1)
        new_images = np.insert(new_images,p,zero145_145,axis = 2)
        for j in range(np.shape(images)[0]):
            for k in range(np.shape(images)[1]):
               if((1<j<143) & (1<k<143)):
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

sio.savemat('./hyperspectral_datas/salina/data/ip_corrected5x5std.mat',{'salinas_corrected':images})
images = sio.loadmat('./hyperspectral_datas/salina/data/ip_corrected5x5std')['salinas_corrected']
labels = sio.loadmat('Indian_pines_gt')['indian_pines_gt']

images = images.reshape((21025,1,448))
labels = labels.flatten()

#part of pre-process, remove class == 0 in data, which means normal grand.
zero = np.where(labels == 0) #17-8=9
no0labels = np.delete(labels,zero)
no0images = np.delete(images,zero,axis=0)
for i in range(no0labels.size):
    no0labels[i] -= 1

num_classes = np.max(no0labels) + 1

#suit the data format
x_train = np.rint(no0images)
x_train = torch.tensor(x_train)
x_train = torchvision.transforms.Normalize(mean = 0, std = 1)(x_train)
y_train = torch.tensor(no0labels)

#combine data and target
train_dataset = TensorDataset(x_train,y_train)

#load data and target
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100, shuffle=True)    

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
        if (i+1) % 30 == 0:
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
    
torch.save(model, 'ipmodel.pkl')
torch.save(model.state_dict(), 'ipparameter.pkl')