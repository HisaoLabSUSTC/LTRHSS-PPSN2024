import os, sys
import numpy as np
import random
import torch.optim as optim
import torch
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'model'))

from model.LTRHSS import LTRHSS
from feature import calc_convergence_fea, calc_diveristy_feature



min_batch_size = 10
selected_num = 10
obj_num = 3
lr = 1e-4
data = [(np.zeros(100,obj_num),list(range(selected_num))) for i in range(10)]

training_data = data[0:int(len(data)*0.8)]
testing_data  = data[int(len(data)*0.8):]
data_size = len(training_data)
ref = np.zeros((obj_num))
model = LTRHSS(data[0][0],selected_num)


for iteration in range(100):
    random.shuffle(training_data)
    # training
    for i in range(data_size//min_batch_size):
        loss = None
        mini_batch = training_data[i*data_size:(i+1)*data_size]
        X = [val[0] for val in mini_batch]
        Y = [val[1] for val in mini_batch]
        
        for k in range(len(X)):
            convergence_feature = calc_convergence_fea(X[k],ref)
            cur_loss = model(X[k],Y[k],convergence_feature)
            loss = cur_loss if loss is None else cur_loss + loss
        loss = loss / len(X)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # testing
    with torch.no_grad():
        loss = None
        X = [val[0] for val in testing_data]
        Y = [val[1] for val in testing_data]
        for k in range(len(X)):
            convergence_feature = calc_convergence_fea(X[k],ref)
            cur_loss = model(X[k],Y[k],convergence_feature)
            loss = cur_loss if loss is None else cur_loss + loss
        loss = loss / len(X)
        print('iteratoin',iteration, 'testing error=',loss)