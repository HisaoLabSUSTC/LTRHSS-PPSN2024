import torch
from model.LTRHSS import LTRHSS
from scipy.io import loadmat
import numpy as np
from feature import calc_convergence_fea
from argparse import ArgumentParser
import h5py
import os
import sys
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import logging
import time
import scipy.io as scio

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# def parse_args():
#     parser = ArgumentParser(description="LTRHSS")
#     parser.add_argument('--gpu_id', type=int, default=0)
#     # Preprocessing
#     parser.add_argument('--num_obj', type=int, default=3, help="Proportion of validation set")
#     parser.add_argument('--selected_num', type=int, default=200, help="Proportion of testing set")
#     # Model
#     parser.add_argument('--dim', type=int, default=64, help="Dimension for embedding")
#     parser.add_argument('--gamma', type=float, default=0.75, help="patience factor")
#     parser.add_argument('--temp', type=float, default=1e-5, help="temperature. how soft the ranks to be")
#     parser.add_argument('--mode', type=str, default='r', help="which loss to use")
#     # Optimizer
#     parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
#     parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay factor")
#     # Training
#     parser.add_argument('--n_epochs', type=int, default=300, help="Number of epoch during training")
#     parser.add_argument('--every', type=int, default=300, help="Period for evaluating during training")
#     # parser.add_argument('--patience', type=int, default=50, help="patience for early stopping")
#     # MOOP
#     parser.add_argument('--type', type=str, default='loss+', choices=['l2', 'loss', 'loss+', 'none'])
#     # Hard setting
#     parser.add_argument('--scale1', type=float, default=1, help="hard setting scale on obj1")

#     return parser.parse_args()

if __name__ == '__main__':

    seed = 12
    random.seed(seed)
    np.random.seed(seed)

    # args = parse_args()
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    obj_num = 8
    lr = 1e-2
    min_batch_size = 64
    run = 12
    data_type = 'train'
    candidate_size = 100
    traindata_num = 10000
    data_name = f'{data_type}_M{obj_num}_dataNum{traindata_num}_k{candidate_size}_run{run}.mat'
    data_path = './data/traindata/'
    path = os.path.join(data_path, data_name)

    dataset = h5py.File(path)
    train_x = np.transpose(dataset.get('Data')) # [dataset_num, candidate_size, M]
    train_y = np.transpose(dataset.get('list_rank')) # [dataset_num, candidate_size]

    data_list = list(range(0, len(train_x)))

    # data = list(zip(train_x, train_y))

    data_size = len(train_x)
    reference_point = np.ones((1, obj_num))*1.1

    model = LTRHSS(device, obj_num)
    # dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=min_batch_size, num_workers=4)

    logger.info('Train begin')
    start_time = time.time()
    train_losses = {'loss': [], 'time': []}

    for iteration in range(100):
        random.shuffle(data_list)
        # training 
        logger.info(f'Epoch {iteration+1}-------------------------------')
        epoch_losses = []
        
        for i in range(data_size//min_batch_size):
            loss = None
            batch_list = data_list[i*min_batch_size:(i+1)*min_batch_size]
            batch_train_x = train_x[batch_list, :, :]
            batch_train_y = train_y[batch_list, :]
            
            for j in range(batch_train_x.shape[0]):
                current_train_x = np.squeeze(batch_train_x[j,:,:])
                current_train_y = batch_train_y[j,:]
                convergence_feautre = calc_convergence_fea(current_train_x, reference_point)
                cur_loss = model(current_train_x, current_train_y, convergence_feautre)
                loss = cur_loss if loss is None else cur_loss + loss
                
            loss = loss / batch_train_x.shape[0]
            optimizer = optim.Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(float(loss))
            
            if (i+1) % 100 == 0:
                logger.info(f'Finshi {(i+1)/100}% batch')
        
        train_losses['loss'].append(np.mean(epoch_losses))
        end_time = time.time()    
        train_losses['time'].append(end_time - start_time)
        logger.info(f"Epoch {iteration+1} -> train loss: {train_losses['loss'][-1]}")
            
    logger.info("Train Done!")

    save_path = './data/trained_model'
    save_model_file = f'trained_model_M{obj_num}_dataNum{traindata_num}_k{candidate_size}_run{run}.pth'
    torch.save(model.state_dict(), os.path.join(save_path, 'model', save_model_file))

    logger.info(f"Saved PyTorch Model State to {save_model_file}")


    # save train_losses
    save_log_file = f'train_log_M{obj_num}_dataNum{traindata_num}_k{candidate_size}_run{run}.mat'
    for key in train_losses.keys():
        train_losses[key] = np.array(train_losses[key])
    scio.savemat(os.path.join(save_path, 'log', save_log_file),
                    train_losses)




# pf_shape = 'Concave'
# candidate_size = 200
# num_objective = 3
# run = 1

# data_name = '{}_N{}_M{}_{}.mat'.format(pf_shape, candidate_size, num_objective, run)
# mat_data = loadmat('./data/candidate/{}'.format(data_name))
# candidates = mat_data['uu'].T




# convergence_fea = calc_convergence_fea(candidates, reference_point)

# y = np.arange(200)

# device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

# loss = model(candidates, y, convergence_fea)


# print('end')