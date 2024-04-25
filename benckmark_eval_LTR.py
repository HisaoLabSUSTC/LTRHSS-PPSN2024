import os
import sys

import numpy as np
import scipy.io as scio
import h5py
import torch
from model.LTRHSS import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
from feature import calc_convergence_fea
import time

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

obj_num = 8
lr = 1e-4
min_batch_size = 100
run = 18
data_type = 'test'
traindata_num = 10000
testdata_num = 100
MaxN = 100
run = 18
N = 100
M = 8
shape_set = ["Concave","Convex","Linear","I-Concave","I-Convex","I-Linear"]
set_index = 1

for ss in np.arange(len(shape_set)):
    shape = shape_set[ss]
    time_all = []
    pred_ranking_all = []
    for ii in np.arange(20)+1:
        set_index = ii
        data_name = f'{shape}_N{N}_M{M}_{set_index}.mat'
        data_path = './data/benchmark_set/'
        path = os.path.join(data_path, data_name)

        dataset = h5py.File(path)
        train_x = dataset.get('uu') # [dataset_num, candidate_size, M]
        
        if train_x is None:
            train_x = dataset.get('uu1') 
        # train_y = np.transpose(dataset.get('list_rank')) # [dataset_num, candidate_size]

        reference_point = np.ones((1, obj_num))*1.1
        # convergence_feautre = calc_convergence_fea(train_x, reference_point)

        data_list = list(range(0, len(train_x)))

        model_file = f'trained_model_M{obj_num}_train_size{traindata_num}_MaxN{MaxN}_run{run}.pth'
        model_data_path = './data/trained_dataset/model'

        model = LTRHSS(device, obj_num)
        model.load_state_dict(torch.load(os.path.join(model_data_path, model_file)))
        model = model.to(device)
        # print(f"Load model {model_file} done!")

        each_train_x = train_x
        # each_train_x = np.squeeze(each_train_x_temp)
        each_convergence_feature = calc_convergence_fea(each_train_x, reference_point)
        start_time = time.time()
        pred_ranking = model.eval_LTR(each_train_x, each_convergence_feature)
        end_time = time.time()
        run_time = end_time - start_time
        time_all.append(run_time)
        # y_true = train_y[i,:]
        pred_ranking_all.append(pred_ranking)
            
    save_path = "./data/predicted_data" 
    save_mat = f'predict_rank_{shape}_N{N}_M{M}.mat'
    pred_ranking_all_dict = {'pred_ranking_all': pred_ranking_all}
    run_time_all_dict = {'runtime': time_all}
    merged_dict = {**pred_ranking_all_dict, **run_time_all_dict}
    scio.savemat(os.path.join(save_path, save_mat), merged_dict)
    
    # scio.savemat(os.path.join(save_path, save_mat), pred_ranking_all_dict, run_time_all_dict)

    print(f"{shape} done!")
    
print(f"all done!")