# -*- coding: utf-8 -*-

#Imports
import torch
from torch.autograd import Variable
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import pickle
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import pyshtools as sht

use_cuda = False#torch.cuda.is_available()
print("Using cuda?: "+str(use_cuda))

mat_size = 30   #30x30 matrix
fill_rat = 0.5  #50% filled, 50% empty
data_amount = 200
data_am_fill = int(fill_rat*data_amount)
data_am_empty = int((1-fill_rat)*data_amount)


def create_data_arr(data_amount_filled = data_am_fill,data_amount_empty = data_am_empty,matrix_size = mat_size):
    '''
    Create array holding a matrix and a type(0,1) -> data, classifier
    '''
    arr = []
    
    for i in range(data_amount_empty):
        mat = np.random.rand(matrix_size,matrix_size)
        arr.append(mat)
    for k in range(data_amount_filled):
        mat = np.ones((matrix_size,matrix_size))+10*np.random.rand(matrix_size,matrix_size)
        #turn half of it into -1
        for x in range(int(matrix_size/2),matrix_size):
            mat[x] -= 2
        arr.append(mat)
    return np.array(arr)


data_array = create_data_arr()

print("Length of data array: "+str(len(data_array)))
random.shuffle(data_array)
# Data created. Split into test/train arr and batches
train_rat = 0.75
max_train_id = train_rat*len(data_array)
print("Max train id: "+str(max_train_id))
train_data_arr = data_array[:int(max_train_id)]
test_data_arr = data_array[int(max_train_id):]
bat_size = 10
#print(len(train_data_arr))
#print(len(test_data_arr))

def arr_to_batches(array, batch_size):
    '''
    Create array filled with batches of data from initial array
    '''
    fin_arr = []
    arr_len = len(array)
    overflow = arr_len%batch_size
    print(str(overflow) + " datapoints too many. Discarding.")
    num_bat = arr_len/batch_size
    print(str(num_bat) + " batches to be made.")
    for i in range(int(num_bat)):
        low_lim = i*batch_size
        up_lim = (i+1)*batch_size
        
        fin_arr.append(array[low_lim:up_lim])
    return np.array(fin_arr)
train_batches = arr_to_batches(train_data_arr,bat_size)
#train_batches now an array of batches(10 tuples per batch)
test_batches = arr_to_batches(test_data_arr,bat_size)
tr_ar = []
te_ar = []
for batch in train_batches:
    tr_m = []
    tr_i = []
    
    for elem in batch:
        tr_m.append(elem)
        idx = 0 if 1 not in elem else 1
        tr_i.append(idx)
    #print("Maps in batch: "+str(len(tr_m)))
    tr_ar.append((np.array(tr_m),np.array(tr_i)))
for batch in test_batches:
    te_m = []
    te_i = []
    for elem in batch:
        te_m.append(elem)
        idx = 0 if 1 not in elem else 1
        #print(idx)
        te_i.append(idx)
    te_ar.append((np.array(te_m),np.array(te_i)))

# Network from original problem
output_size = 2
    
img_size = mat_size
    
in_channels = 1
out_channels_conv1 = 1
kernel_conv1 = 3  #Conv layer 1
out_channels_conv2 = 1
kernel_conv2 = 2   #Conv layer 2
pooling_kernel_1 = 2 #Pooling layer 1
pooling_kernel_2 = 2  #Pooling layer 2
lin_input_size = in_channels*(int((((img_size - kernel_conv1 -1)/pooling_kernel_1)-kernel_conv2-1)/pooling_kernel_2)+1)**2
lin_output_size = 2


class network(nn.Module):
    
    #cifar
    def __init__(self):
        super(network, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(10, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
       
        return x
    '''
    #mnist
    def __init__(self):
        super(network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
    def __init__(self,batch_size = bat_size,num_classes = 2,   #Data for input/output of lin layer
                 in_channels = in_channels,out_channels_conv1 = out_channels_conv1,kernel_conv1 = kernel_conv1,  #Conv layer 1
                 out_channels_conv2 = out_channels_conv2,kernel_conv2 = kernel_conv2,   #Conv layer 2
                 pooling_kernel_1 = pooling_kernel_1, #Pooling layer 1
                 pooling_kernel_2 = pooling_kernel_2,  #Pooling layer 2
                 lin_input_size = lin_input_size,lin_output_size = lin_output_size): #lin layer
        
        #define neural network
        
        super(network,self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.con1 = nn.Conv2d(in_channels,out_channels_conv1,kernel_conv1)
        self.pool1 = nn.MaxPool2d(pooling_kernel_1)
        self.con2 = nn.Conv2d(out_channels_conv1,out_channels_conv2,kernel_conv2)
        self.pool2 = nn.MaxPool2d(pooling_kernel_2)
        self.lin = nn.Linear(lin_input_size,lin_output_size)
        
    def forward(self,x):
        batsize = x.data.shape[0]
        x = self.con1(F.sigmoid(x))#F.sigmoid(self.con1(x))#self.con1(F.sigmoid(x))
        x = self.pool1(F.relu(x))#F.relu(self.pool1(x))#self.pool1(F.relu(x))
        x = self.con2(F.relu(x))#F.relu(self.con2(x))#self.con2(F.relu(x))
        x = self.pool2(F.relu(x))# F.relu(self.pool2(x))#self.pool2(F.relu(x))
        x = x.view(batsize,-1)
        #arg = F.softmax(x,dim = 1)
        x = F.softmax(self.lin(x),dim=1)
        return x
    '''
net = network()
if use_cuda:
    net = net.cuda()
print("="*20)
print(net)
print("="*20)

crit = nn.CrossEntropyLoss()
lr = 1e-5
momentum = 0.9
opti = optim.SGD(net.parameters(),lr,momentum)
epochs = 50

train_losses = []
test_losses = []
correctness = []

def train(epoch):
    running_loss = 0
    net.train()
    random.shuffle(test_batches)
    for batch_id in range(len(tr_ar)):
    
        batch = tr_ar[batch_id]
    
        maps = batch[0]
        idx = np.array([batch[1]])
        maps,idx = Variable(torch.from_numpy(maps)),Variable(torch.from_numpy(idx))
        if use_cuda:
            maps,idx = maps.cuda(),idx.cuda()
        maps = maps.unsqueeze(1)
        idx = idx.squeeze(0)
       
        maps = maps.float()
        #print(maps.shape)
        pred = net(maps)
        #print(pred)
        opti.zero_grad()
        #print(idx)
        loss = crit(pred.float(),idx.long())
        loss.backward()
        '''
        for param in net.parameters():
            print(param.grad.data.sum())

        # start debugger
        import pdb; pdb.set_trace()
        '''
        opti.step()
        running_loss += loss.item()
        if int(batch_id/len(train_batches)*100) % 25 == 0:
            print("[Epoch: "+str(epoch)+"("+str(epoch/max((epochs-1),1)*100)+"%): Data: "+str(batch_id/len(train_batches)*100)+"%]:Running loss: "+str(running_loss))
    train_losses.append(running_loss)
def test(epoch):
     #Run testing for monitoring
    
    net.eval()
    test_loss_train = 0
    correct = 0
    total_test = 0
    for batch_id in range(len(te_ar)):
        batch = te_ar[batch_id]
        cur_maps = batch[0]
        idx = [batch[1]]
        
        in_map = Variable(torch.from_numpy(cur_maps))
            #print("Classifier: ")
            
        classif = Variable(torch.from_numpy(np.array(idx)))
            #print(classif)
        if use_cuda:
            in_map = in_map.cuda()
            classif = classif.cuda()
        in_map = in_map.unsqueeze(1)
        in_map = in_map.float()
        classif = classif.squeeze(0)
        pred = net(in_map)
        loss = crit(pred.float(),classif.long())
        test_loss_train += loss.item()
        classif = classif.long()
        pred_class = pred.data.max(1,keepdim = True)[1] #max index
        pred_class = pred_class.long()
            
        correct += pred_class.eq(classif.data.view_as(pred_class)).long().cpu().sum()
        total_test += len(batch[0])
        
            #print("Correctness Values: ")
            
            #print(pred_class.eq(classif.data.view_as(pred_class)).long())
    print("Test set accuracy: "+str(100*correct.item()/total_test) + "% ,loss = "+str(test_loss_train))
    print("Correct hits: "+str(correct.item()))
    print("Total amount of maps: "+str(total_test))
    #print("Total amount expected: "+str((len(test_batches)*len(test_batches[0][0]))))
    correctness.append(100*correct/total_test)    
    test_losses.append(test_loss_train)
fig, (ax1,ax2,ax3) = plt.subplots(3,1)
for epoch in range(epochs):
    train(epoch)
    test(epoch)
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.plot(train_losses)
    ax1.set_title("Train losses")
    ax2.plot(test_losses)
    ax2.set_title("Test losses")
    ax3.plot(correctness)
    ax3.set_title("Accuracy")
    fig.canvas.set_window_title(str(100*epoch/epochs)+"%")
    plt.pause(1e-8)


