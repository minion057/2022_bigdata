#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as T

import os
import random

# from models import *
# from loader import Loader, RotationLoader
# from utils import progress_bar
import numpy as np

import string
import glob 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import IPython.display as ipd


# # 1 Setting

# ## 1.1 Argument

# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# In[3]:


best_acc = 0 # best test acc
start_epoch = 0 # start from epoch 0 OR checkpoint epoch

learning_rate = 4e-5 #1e-4


# ## 1.2 Data

# In[5]:


PATH_SAVE_DATASET = 'Large_Captcha_4e'


# In[5]:


all_letters = string.ascii_lowercase + string.digits + string.ascii_uppercase

mapping = {}      # key - num    & value - letter
mapping_inv = {}  # key - letter & value - num

for i, x in enumerate(all_letters):
    mapping[x] = i+1
    mapping_inv[i+1] = x

num_class = len(mapping)
num_class


# In[6]:


class Loader_Captcha_active(Dataset):
    def __init__(self, is_train=True, transform=None, path_list=None):
        self.is_train = is_train
        self.transform = transform
        if path_list is None:
            raise Exception("not have a path_list")
        self.img_path = path_list

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        #img = cv2.imread(self.img_path[idx])
        #img = Image.fromarray(img)
        img = Image.open(self.img_path[idx]).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        
        img_name = self.img_path[idx].split('/')[-1]
        label = torch.IntTensor([mapping[i] for i in img_name.split('.')[0]]) #int(self.img_path[idx].split('/')[-1])
        
        return img, label


# In[7]:


transform = T.Compose([
    T.ToTensor()
])


# In[8]:


import pickle
testset = []
with open('./testset.pkl', 'rb') as f:
    testset = pickle.load(f)
len(testset)


# In[9]:


# random.shuffle(testset)
testloader  = torch.utils.data.DataLoader(testset[:1000],  batch_size=10,  shuffle=False)


# ## 1.2 Model

# In[10]:


class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(inp, hidden, bidirectional=True)
        else:
            self.rnn = nn.GRU(inp, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden*2, out)
    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)     
        return out


# In[11]:


class CRNN(nn.Module):
    def __init__(self, in_channels, output):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 256, 9, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(3, 3),
                nn.Conv2d(256, 256, (4, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256))
        
        self.linear = nn.Linear(20992, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.rnn = Bidirectional(256, 1024, output+1)

    def forward(self, X, y=None, criterion = None): # y is target.
        out = self.cnn(X)
        N, C, w, h = out.size()
        out = out.view(N, -1, h)
        out = out.permute(0, 2, 1)
        out = self.linear(out)

        out = out.permute(1, 0, 2)
        out = self.rnn(out)
            
        if y is not None:
            T = out.size(0)
            N = out.size(1)
        
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            target_lengths = torch.full(size=(N,), fill_value=5, dtype=torch.int32)
        
            loss = criterion(out, y, input_lengths, target_lengths)
            
            return out, loss
        
        return out, None
    
    def _ConvLayer(self, inp, out, kernel, stride, padding, bn=False):
        if bn:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU(),
                nn.BatchNorm2d(out)
            ]
        else:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU()
            ]
        return nn.Sequential(*conv)


# In[12]:


model = CRNN(in_channels=1, output=num_class)
model = model.to(device)
# print(net)


# In[13]:


if device == 'cuda':
    #net = torch.nn.DataParallel(net) # 문현안쓸때만쓰기... 나는 0
    cudnn.benchmark = True


# # 2 Training

# In[14]:


def predict(outputs):
    result = []
    for i in range(len(outputs)):
        pred = []
        then = 0
        for x in outputs[i]:
            if then != x and x > 0 :
                pred.append(x)
                if len(pred) == 5:
                    break
            then = x
        if len(pred) < 5:
            for i in range(5-len(pred)):
                pred.append(0)
        result.append(pred)
    result = torch.LongTensor(result).cuda()
    return result


# In[15]:


def train(model, criterion, optimizer, epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0    
    
    tk = tqdm(trainloader, total=len(trainloader))
    #for batch_idx, (inputs, targets) in enumerate(trainloader):
    for inputs, targets in tk:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        out, loss = model(inputs, targets, criterion=criterion)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predicted = predict(out.permute(1, 2, 0).max(1)[1])
        total += targets.size(0) #* 5
        for i in range(len(predicted)):
            correct += torch.equal(predicted[i], targets[i])
#         correct += predicted.eq(targets).sum().item()
        
        tk.set_postfix({'Train - Loss' : loss.item(), '& ACC':100.*correct/total})
    
    return 100.*correct/total, train_loss/total


# # 3 Test

# In[16]:


def test(model, criterion, epoch, cycle):
    global best_acc, EPOCH
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        tk = tqdm(testloader, total=len(testloader))
        for inputs, targets in tk:
            inputs, targets = inputs.to(device), targets.to(device)
            out, loss = model(inputs, targets, criterion=criterion)
            
            test_loss += loss.item()
            predicted = predict(out.permute(1, 2, 0).max(1)[1])
            total += targets.size(0)         
            for i in range(len(predicted)):
                correct += torch.equal(predicted[i], targets[i])

            tk.set_postfix({'Test - Loss' : loss.item(), '& ACC':100.*correct/total})

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # save rotation weights
        torch.save(state, './checkpoint/1_active_learning_'+PATH_SAVE_DATASET+'_'+str(cycle)+'.pth')
        best_acc = acc
        
    if epoch+1 == EPOCH and not os.path.isfile( './checkpoint/1_active_learning_'+PATH_SAVE_DATASET+'_'+str(cycle)+'.pth'):
        print('Saving.. best acc is zero....')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # save rotation weights
        torch.save(state, './checkpoint/1_active_learning_'+PATH_SAVE_DATASET+'_'+str(cycle)+'.pth')
        
    
    return acc, test_loss/total


# # 4 Run

# In[17]:


def save_acc_and_loss(cycle, epoch, acc, loss, is_train = True):
    task_name = 'epoch - '+str(epoch)+'\ntrain > ' if is_train else 'test > '
    if epoch == 0 and is_train:
        task_name = 'Cycle '+str(cycle)+' ====================== \n'+task_name
    file_path = './1_active_learing_result_'+PATH_SAVE_DATASET+'.txt'
    with open(file_path,'a') as f:
        f.write(task_name + 'acc_' + str(acc) + ' & loss_' + str(loss) +'\n')


# In[18]:


# confidence sampling (pseudo labeling)
## return 1k samples w/ lowest top1 score
def get_plabels2(net, samples, cycle, slice_num):
    sub_dataset = Loader_Captcha_active(is_train=False,  transform=transform, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub_dataset, batch_size=1, shuffle=False)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        tk = tqdm(ploader, total=len(ploader))
        for inputs, targets in tk:
            inputs, targets = inputs.to(device), targets.to(device)
            out, loss = net(inputs, targets, criterion=nn.CTCLoss())
            predicted = predict(out.permute(1, 2, 0).max(1)[1])
            
            # save top1 confidence score 
            outputs = F.normalize(out.permute(1, 2, 0), dim=1)
            probs = F.softmax(out.permute(1, 2, 0), dim=1)
            
            # 원래는 예측한 class probs 값을 score로 사용함
            # 하지만 우리의 label은 5개로 예측값도 5개, prob도 5개임
            # 따라서 5개의 예측한 class의 prob를 평균내고 그 중 max값 활용
            # our ouput num is 5 >> 예측 확률 mean >> max  (가장 높은 값을 뽑아 낮은 data로 정렬) 
            top1_scores.append(np.mean(probs[0][predicted.cpu()].tolist()[0], axis=0).max())
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[:slice_num]]


# In[19]:


def main(slice_num, epoch, cycles, start):
    labeled=[]    
    global history_train, history_test, best_acc# 여기서 best acc 지정해주기!!!
    
    model = CRNN(in_channels=1, output=num_class)
    model = model.to(device)
    criterion = nn.CTCLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    
    for cycle in range(start, cycles):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])
        best_acc = 0
        
        # open mini batch (sorted low->high)
        with open('./loss_Large_Captcha_5000/batch_' + str(cycle) + '.txt', 'r') as f:
            samples = f.readlines()[:-2]
        samples = [s_path[:-1] for s_path in samples] #  \n 지우기

        if cycle > 0:
            print('>> Getting previous checkpoint')
            # './checkpoint/1_active_learning_'+PATH_SAVE_DATASET+str(cycle)+'.pth'
            checkpoint = torch.load(f'./checkpoint/1_active_learning_{PATH_SAVE_DATASET}_{cycle-1}.pth')
            net = CRNN(in_channels=1, output=num_class)
            net = net.to(device)
            net.load_state_dict(checkpoint['model'])
            # sampling
            sample_sort = get_plabels2(net, samples, cycle, slice_num)
        else:
            samples = np.array(samples)
            random.shuffle(samples)
            sample_sort = samples[:slice_num]
        labeled.extend(sample_sort)
        print(f'>> Labeled length: {len(labeled)}')
        activeset = Loader_Captcha_active(is_train=False,  transform=transform, path_list=labeled)
        trainloader = torch.utils.data.DataLoader(activeset, batch_size=32, shuffle=True)

        model = CRNN(in_channels=1, output=num_class)
        model = model.to(device)
        criterion = nn.CTCLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for e in range(epoch):
            ipd.clear_output(wait=True)
            print('Cycle ', cycle)
            print(f'>> Labeled length: {len(labeled)}')

            acc, loss = train(model, criterion, optimizer, e, trainloader)
            history_train['acc'][cycle].append(acc)
            history_train['loss'][cycle].append(loss)
            save_acc_and_loss(cycle, e, acc, loss, is_train = True)

            acc, loss = test(model, criterion, e, cycle)
            history_test['acc'][cycle].append(acc)
            history_test['loss'][cycle].append(loss)
            save_acc_and_loss(cycle, e, acc, loss, is_train = False)
            
            scheduler.step()
        with open(f'./1_active_learning_main_best_'+PATH_SAVE_DATASET+'.txt', 'a') as f:
            f.write(str(cycle) + ' ' + str(best_acc)+'\n')


# In[20]:


EPOCH= 100
START_cycles = len(glob.glob(f'./checkpoint/1_active_learning_{PATH_SAVE_DATASET}_*.pth'))
CYCLES = len(glob.glob('./loss_Large_Captcha_5000/batch_*.txt'))


# In[21]:


history_train = {'acc':[[] for i in range(CYCLES)], 'loss':[[] for i in range(CYCLES)]} 
history_test  = {'acc':[[] for i in range(CYCLES)], 'loss':[[] for i in range(CYCLES)]}


# In[ ]:


main(slice_num=500, epoch=EPOCH, cycles=CYCLES, start = START_cycles)


# In[ ]:




