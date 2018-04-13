"""
Dependencies:
torch: 0.1.11
matplotlib
torchvision
"""
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import h5py
import numpy as np


#  make UCF101 feature dataset
def make_dataset(pwd, dim, ft_layer):
  f_list = open(pwd)
  da1 = []
  da2 = []
  da3 = []
  la = []

  for line in f_list:
    f = h5py.File(line[:-1],'r') 

    d1_numpy = np.array(f['data'][0][:24])
    d2_numpy = np.array(f['data'][0][1:])
    d3_numpy = np.array(f['data'][0][25-ft_layer:])
    l_numpy = np.array(f['label'])
    da1.append(d1_numpy)
    da2.append(d2_numpy)
    da3.append(d3_numpy)
    la.append(l_numpy)
  
    f.close()
    #break
  da1_re = np.reshape(np.array(da1), (dim,24,2048))
  da2_re = np.reshape(np.array(da2), (dim,24,2048))
  da3_re = np.reshape(np.array(da3), (dim,ft_layer,2048))
  la_re = np.reshape(np.array(la), dim)
  data_raw = torch.from_numpy(da1_re)
  data_predict = torch.from_numpy(da2_re)
  data_finetune = torch.from_numpy(da3_re)
  label = torch.from_numpy(la_re)
  label = label.long()
  print data_raw.size()
  print data_predict.size()
  print data_finetune.size()
  print label.size()
  f_list.close()
  return data_raw,data_predict,data_finetune,label

ft_layer = 2

train_pwd = '/disk_new/liuyj/FC6_trainlist25p03/fc6list_pool5_resnet_rgb.txt'
test_pwd = '/disk_new/liuyj/FC6_testlist25p03/fc6list_pool5_resnet_rgb.txt'
train_data = make_dataset(train_pwd, 9624, ft_layer)
test_data = make_dataset(test_pwd, 3696, ft_layer)



class MyDataset():
  def __init__(self, images, labels):
    self.images = images
    self.labels = labels

  def __getitem__(self, index):
    img, target = self.images[index], self.labels[index]
    return img, target

  def __len__(self):
    return len(self.images)

train_dataset = MyDataset(images=train_data[0], labels=train_data[1])
train_dataset_ft = MyDataset(images=train_data[2], labels=train_data[3])



# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 60               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 10 #10
TIME_STEP = 24         # rnn time step / image height
INPUT_SIZE = 2048         # rnn input size / image width
LR = 0.1               # learning rate
#DOWNLOAD_MNIST = False   # set to True if haven't download the data


# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_ft = torch.utils.data.DataLoader(dataset=train_dataset_ft, batch_size=BATCH_SIZE, shuffle=True)
#print train_loader.size()

# convert test data into Variable, pick 2000 samples to speed up testing

test_x_cpu = Variable(test_data[0], volatile=True).type(torch.FloatTensor)[:2000]  # shape (2000, 28, 28) value in range(0,1)
test_y = test_data[1].numpy().squeeze()[:2000]    # covert to numpy array
test_x = test_x_cpu.cuda()

test_x_cpu_ft = Variable(test_data[2], volatile=True).type(torch.FloatTensor)[:2000]  # shape (2000, 28, 28) value in range(0,1)
test_y_ft = test_data[3].numpy().squeeze()[:2000]    # covert to numpy array
test_x_ft = test_x_cpu.cuda()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=1024,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out1 = nn.Linear(1024, 2048)	#output 2048 dim feature
        self.out2 = nn.Linear(1024, 101)


    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out1 = self.out1(r_out[:, :, :])
        out2 = self.out2(r_out[:, -1, :])

        return out1,out2


rnn = RNN()
print(rnn)

######### forward or pre-train

optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)   # optimize all cnn parameters
#loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
loss_func = nn.MSELoss() 				#MSE loss
rnn.cuda()
loss_func.cuda()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# training and testing
for epoch in range(1):
    for step, (x, y) in enumerate(train_loader):        # gives batch data
        #b_x_cpu = Variable(x.view(-1, 25, 2048))              # reshape x to (batch, time_step, input_size)
        b_x_cpu = Variable(x.view(-1, 24, 2048))
        b_x = b_x_cpu.cuda()

        b_y_cpu = Variable(y.view(-1, 24, 2048))                               # batch y
        b_y = b_y_cpu.cuda()

        output = rnn(b_x)[0]                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            test_output_gpu = rnn(test_x)[0]                   # (samples, time_step, input_size)
            test_output = test_output_gpu.cpu()
            pred_y_gpu = torch.max(test_output, 1)[1].cuda().data.squeeze()
            pred_y = pred_y_gpu.cpu().numpy()
            #accuracy = sum(pred_y == test_y) / float(test_y.size)
            accuracy = 0
            print('Epoch: ', epoch, '| train loss: %.8f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

######### backward or finetune
optimizer = torch.optim.SGD(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
#loss_func = nn.MSELoss() 				#MSE loss
rnn.cuda()
loss_func.cuda()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader_ft):        # gives batch data
        #b_x_cpu = Variable(x.view(-1, 25, 2048))              # reshape x to (batch, time_step, input_size)
        b_x_cpu = Variable(x.view(-1, ft_layer, 2048))
        b_x = b_x_cpu.cuda()

        b_y_cpu = Variable(y)                               # batch y
        b_y = b_y_cpu.cuda()

        output = rnn(b_x)[1]                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 10 == 0:
            test_output_gpu = rnn(test_x)[1]                   # (samples, time_step, input_size)
            test_output = test_output_gpu.cpu()
            pred_y_gpu = torch.max(test_output, 1)[1].cuda().data.squeeze()
            pred_y = pred_y_gpu.cpu().numpy()
            accuracy = sum(pred_y == test_y_ft) / float(test_y_ft.size)
            #accuracy = 0
            print('Epoch: ', epoch, '| train loss2: %.8f' % loss.data[0], '| test accuracy2: %.4f' % accuracy)


# print 10 predictions from test data

test_output = rnn(test_x[:10].view(-1, 24, 2048))[1]
pred_y_gpu = torch.max(test_output, 1)[1].cuda().data.squeeze()
pred_y = pred_y_gpu.cpu().numpy()
print(pred_y, 'prediction number')
print(test_y_ft[:10], 'real number')

