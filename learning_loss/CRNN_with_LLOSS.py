#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import glob 
import pandas as pd
import string
import collections

from tqdm import tqdm

from PIL import Image

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import models.lossnet as lossnet
import models.crnn as crnn
from config import *
from sampler import SubsetSequentialSampler


# In[2]:


data = glob.glob(os.path.join('./Large_Captcha_Dataset', '*.png'))
path = './Large_Captcha_Dataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data.remove(os.path.join('./Large_Captcha_Dataset', '4q2wA.png'))


# In[3]:


all_letters = string.ascii_uppercase + string.ascii_lowercase + string.digits

mapping = {}
mapping_inv = {}
i = 1
for x in all_letters:
    mapping[x] = i
    mapping_inv[i] = x
    i += 1

num_class = len(mapping)


# In[4]:


images = []
labels = []
datas = collections.defaultdict(list)
for d in data:
    x = d.split('/')[-1]
    datas['image'].append(x)
    datas['label'].append([mapping[i] for i in x.split('.')[0]])
df = pd.DataFrame(datas)
df.head()


# In[5]:


class CaptchaDataset:
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image = Image.open(os.path.join(path, data['image'])).convert('L')
        label = torch.tensor(data['label'], dtype=torch.int32)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


# In[6]:


df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)

train_transform = T.Compose([
    # T.RandomHorizontalFlip(),
    # T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914,], [0.2023,]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914,], [0.2023,]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

trainset = CaptchaDataset(df_train, train_transform)
unlabeledset = CaptchaDataset(df_train, test_transform)
testset = CaptchaDataset(df_test, test_transform)

# trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
# testloader = DataLoader(testset, batch_size=32)


# In[7]:


# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


# In[8]:


def get_uncertainty(models, criterion, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, features, _ = models['backbone'](inputs, labels, criterion)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()


# In[9]:


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


# In[10]:


# Train Utils
iters = 0

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    models['backbone'].train()
    models['module'].train()
    global iters

    # loss_cum_1, loss_cum_2 = 0, 0

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features, target_loss = models['backbone'](inputs, labels, criterion)
        # target_loss = target_loss.log_softmax(-1)
        # target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            # features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        loss            = m_backbone_loss + WEIGHT * m_module_loss
        # loss_cum_1 += m_backbone_loss
        # loss_cum_2 += m_module_loss
        # print(f'm_backbone_loss : {m_backbone_loss}, m_module_loss : {m_module_loss}, loss : {loss}')

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

    # return loss_cum_1, loss_cum_2
        # Visualize
        # if (iters % 100 == 0) and (vis != None) and (plot_data != None):
        #     plot_data['X'].append(iters)
        #     plot_data['Y'].append([
        #         m_backbone_loss.item(),
        #         m_module_loss.item(),
        #         loss.item()
        #     ])
        #     vis.line(
        #         X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
        #         Y=np.array(plot_data['Y']),
        #         opts={
        #             'title': 'Loss over Time',
        #             'legend': plot_data['legend'],
        #             'xlabel': 'Iterations',
        #             'ylabel': 'Loss',
        #             'width': 1200,
        #             'height': 390,
        #         },
        #         win=1
        #     )


# In[11]:


def test(models, criterion, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[mode]):
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            # print(idx)
            # print(f'!!!!!!!!!!!! input size : {inputs.size()} label size : {labels.size()}')
            scores, _, _ = models['backbone'](inputs, labels, criterion)
            scores = scores.permute(1, 0, 2)
            _, preds = torch.max(scores.data, 2)
            # print(f'!!!!!!!!!!!! previous pred : {preds}')
            preds = predict(preds)
            # print(f'!!!!!!!!!!!! pred size : {preds.size()}, labels : {labels.size()}, pred : {preds}, labels : {labels}')
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if idx == 100:
                break
    
    return 100 * correct / total


# In[12]:


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis, plot_data):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./checkpoints', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        # schedulers['backbone'].step()
        # schedulers['module'].step()

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis, plot_data)
        # print(f'loss_cum_1 : {loss_cum_1}, loss_cum_2 : {loss_cum_2}')

        # Save a checkpoint
        # if True and epoch % 4 == 0:
            # acc = test(models, criterion, dataloaders, 'test')
            # if best_acc < acc:
            #     best_acc = acc
            #     torch.save({
            #         'epoch': epoch + 1,
            #         'state_dict_backbone': models['backbone'].state_dict(),
            #         'state_dict_module': models['module'].state_dict()
            #     },
            #     '%s/crnn_lloss.pth' % (checkpoint_dir))
            # print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')


# In[13]:


vis = None
plot_data = None

for trial in range(TRIALS):
    # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
    indices = list(range(NUM_TRAIN))
    random.shuffle(indices)
    labeled_set = indices[:ADDENDUM]
    unlabeled_set = indices[ADDENDUM:]
    
    train_loader = DataLoader(trainset, batch_size=BATCH, 
                                sampler=SubsetRandomSampler(labeled_set), 
                                pin_memory=True)
    test_loader  = DataLoader(testset, batch_size=BATCH)
    dataloaders  = {'train': train_loader, 'test': test_loader}
    
    # Model
    crnn_model    = crnn.CRNN(in_channels=1, output=num_class).cuda()
    loss_module = lossnet.LossNet().cuda()
    models      = {'backbone': crnn_model, 'module': loss_module}
    torch.backends.cudnn.benchmark = False

    # Active learning cycles
    for cycle in range(CYCLES):
        # Loss, criterion and scheduler (re)initialization
        criterion      = nn.CTCLoss(reduction='none') # nn.CrossEntropyLoss(reduction='none')
        optim_backbone = optim.Adam(models['backbone'].parameters(), lr=LR,
                                weight_decay=WDECAY)
        # optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
        #                         momentum=MOMENTUM, weight_decay=WDECAY)
        optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                                momentum=MOMENTUM, weight_decay=WDECAY)
        # sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
        # sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

        optimizers = {'backbone': optim_backbone, 'module': optim_module}
        schedulers = None
        # schedulers = {'backbone': sched_backbone, 'module': sched_module}

        # Training and test
        train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data)
        acc = test(models, criterion, dataloaders, mode='test')
        print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))

        ##
        #  Update the labeled dataset via loss prediction-based uncertainty measurement

        # Randomly sample 10000 unlabeled data points
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:SUBSET]

        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(unlabeledset, batch_size=BATCH, 
                                        sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                        pin_memory=True)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty(models, criterion, unlabeled_loader)

        # Index in ascending order
        arg = np.argsort(uncertainty)
        
        # Update the labeled dataset and the unlabeled dataset, respectively
        labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
        unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

        # Create a new dataloader for the updated labeled dataset
        dataloaders['train'] = DataLoader(trainset, batch_size=BATCH, 
                                            sampler=SubsetRandomSampler(labeled_set), 
                                            pin_memory=True)
    
    # Save a checkpoint
    torch.save({
                'trial': trial + 1,
                'state_dict_backbone': models['backbone'].state_dict(),
                'state_dict_module': models['module'].state_dict()
            },
            './checkpoints/trial/weights/crnn_lloss_{}.pth'.format(trial))


# In[ ]:


# model = crnn.CRNN(in_channels=1, output=num_class).to(device)
# loss_pred_module = lossnet.LossNet().to(device)

# criterion = nn.CTCLoss(reduction='none')
# loss_pred_criterion = MarginRankingLoss_learning_loss()

# optimizer_target = optim.Adam(model.parameters(), lr=1e-1)
# optimizer_loss = optim.SGD(loss_pred_module.parameters(), lr=1e-1)


# In[ ]:


# def predict(outputs):
#     result = []
#     for i in range(32):
#         pred = []
#         then = 0
#         for x in outputs[i]:
#             if then != x and x > 0 :
#                 pred.append(x)
#                 if len(pred) == 5:
#                     break
#             then = x
#         if len(pred) < 5:
#             for i in range(5-len(pred)):
#                 pred.append(0)
#         result.append(pred)
#     result = torch.LongTensor(result).to(device)
#     return result


# In[ ]:


# # Training
# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     model.train()
#     loss_pred_module.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     pbar = tqdm(trainloader, total=len(trainloader))
#     for data in pbar:
#         inputs, targets = data[0].to(device), data[1].to(device)
#         optimizer_target.zero_grad()
#         optimizer_loss.zero_grad()

#         outputs, loss_pred, loss = model(inputs, targets, criterion=criterion)
#         loss_pred = loss_pred_module(loss_pred)
#         loss_prediction_loss = loss_pred_criterion(loss_pred, loss)
#         target_loss = loss.mean()
#         if epoch < 120:
#             loss = loss_prediction_loss + target_loss 
#             loss.backward()
#             optimizer_target.step()
#             optimizer_loss.step()
#         else:
#             loss = target_loss
#             loss.backward()
#             optimizer_target.step()

#         train_loss = loss.item()
#         # _, predicted = outputs.max(1)


#         outputs = outputs.permute(1, 0, 2)
#         # outputs = outputs.log_softmax(2)
#         outputs = outputs.argmax(2)
#         # outputs = outputs.cpu().detach().numpy()

#         print(outputs.size())
#         result = predict(outputs)
        
#         total += targets.size(0)
#         # if result[0]:
#         correct += result.eq(targets).sum().item()
#         pbar.set_description(f'Loss: {train_loss:.3f} | Acc: {100.*correct/total:.3f} ({correct}/{total})')


# In[ ]:


# def test(epoch):
#     global best_acc
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         pbar = tqdm(testloader, total=len(testloader))
#         for data in pbar:
#             inputs, targets = data[0].to(device), data[1].to(device)
#             outputs, loss_pred, loss = model(inputs, targets, criterion=criterion)
#             loss = loss.mean()
#             test_loss = loss.item()
            
#             # _, predicted = outputs.max(1)
#             # total += targets.size(0)
#             # correct += predicted.eq(targets).sum().item()

#             outputs = outputs.permute(1, 0, 2)
#             # outputs = outputs.log_softmax(2)
#             outputs = outputs.argmax(2)

#             result = predict(outputs)

#             total += targets.size(0)
#             correct += result.eq(targets).sum().item()

    
#         # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': model.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.pth')
#         best_acc = acc


# In[ ]:


# for epoch in range(0, 5):
#     train(epoch)
#     test(epoch)


# In[ ]:




