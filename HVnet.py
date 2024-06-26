import numpy as np
import scipy.io as scio
import h5py
import torch
from models import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random

import sys
import os
import tqdm
import time

import logging


def setup_logging(log_name=__name__, level=logging.INFO, log_file='train_log.log'):
    """
    return a logger that handler for StreamHandler and FileHandler
    Args:
        log_name: usually to be the file name: __name__
        level: set record level DEBUG < INFO < WARNING < ERROR < CRITICAL
        log_file: file name for log file

    Returns: logger instance

    """
    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


# CMD code: bash run.sh 1 train_data_M8_2.mat model_M8_MSE_2.pth
if __name__ == '__main__':
    path_dir = '/liaoweiduo/HV-Net-datasets'  # for ubuntu server
    # path_dir = '//10.20.2.245/datasets/HV-Net-datasets'  # for windows

    if len(sys.argv) == 3:
        train_file = sys.argv[1]
        save_file = sys.argv[2]
    else:
        train_file = 'train_data_whole_1.mat'
        save_file = 'model_whole_1.pth'

    log_file_name = f"train_log_{train_file[:-4]}_{save_file[:-4]}"

    batch_size = 100
    epochs = 100

    path = os.path.join(path_dir, 'data', train_file)
    # data = scio.loadmat(path)
    data = h5py.File(path)

    log_path = os.path.join(path_dir, "log", f"{log_file_name}.log")
    logger = setup_logging(__name__, logging.DEBUG,
                           log_path)
    logger.info(f"training data path: {path}")
    logger.info(f"training log path: {log_path}")

    solutionset = torch.from_numpy(np.transpose(data.get('Data'))).float()      # [dataset_num, data_num, M]
    hv = torch.from_numpy(np.transpose(data.get('HVval'))).float()              # [dataset_num, 1]
    hv = torch.reshape(hv, (hv.shape[0],1,1))                                   # [dataset_num, num_outputs, dim_output]
    logger.info(f"solution set shape: {solutionset.shape}")
    logger.info(f"hv shape: {hv.shape}")

    dim_input = solutionset.shape[2]        # M=3
    num_outputs = 1
    dim_output = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    settransformer = DeepSet(device, dim_input, num_outputs, dim_output)
    #settransformer = SetTransformer(dim_input, num_outputs, dim_output)

    ## reload and further train
    # settransformer.load_state_dict(torch.load(os.path.join(path_dir, save_file)))
    # print("Load Done!")

    settransformer = settransformer.to(device)

    optimizer = torch.optim.Adam(settransformer.parameters(), lr=1e-4)
    #optimizer = torch.optim.SGD(settransformer.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction='mean')


    ## Train
    logger.info('Train')
    #deepset.train()
    settransformer.train()

    size = solutionset.shape[0]
    train_losses = {'loss': [], 'time': []}
    start_time = time.time()
    with tqdm.tqdm(total=epochs*int(size/batch_size), file=sys.stdout) as pbar_train:
        for t in range(epochs):
            logger.info(f"Epoch {t+1}-------------------------------")
            epoch_losses = []
            dataloader = DataLoader(TensorDataset(solutionset, hv), batch_size=batch_size, shuffle=True, num_workers=10)
            for batch, (X, y) in enumerate(dataloader):             # all data train once in 1 epoch.
                X, y = X.to(device), y.to(device)   # [bs, 100, 3] [bs, 1, 1]
                ## Compute prediction error
                loss = []
                for batch_idx in range(batch_size):
                    input, output = X[batch_idx:batch_idx + 1], y[batch_idx:batch_idx + 1]      # [1, 100, 3] [1, 1, 1]
                    mask = ~torch.isnan(input[0, :, 0])     # [100]
                    input = input[:, mask == True]          # [1, 30, 3]
                    pred = settransformer(input)            # [1, 1, 1]
                    # loss.append(loss_fn(pred, output))                          # MSE
                    loss.append(loss_fn(torch.log(pred), torch.log(output)))    # MSE_log
                    # loss.append(torch.mean(torch.abs(pred, output)/output))                          # relative MAE

                loss = torch.mean(torch.stack(loss))

                # pred = settransformer.forward_allow_nan(input)             # [bs, num_outputs, dim_output]  [bs,1,1]
                # loss = loss_fn(torch.log(pred), torch.log(output))

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(float(loss))
                train_output_update = f'loss: {float(loss):>7f}'

                pbar_train.update(1)
                pbar_train.set_description(f"training phase {t+1} -> {train_output_update}")

                # if batch % 5 == 0:
                #     loss, current = loss.item(), batch * len(X)
                #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            train_losses['loss'].append(np.mean(epoch_losses))
            end_time = time.time()
            train_losses['time'].append(end_time - start_time)
            logger.info(f"Epoch {t+1} -> train loss: {train_losses['loss'][-1]}")

        ## Val
        #TBD

    logger.info("Train Done!")

    torch.save(settransformer.state_dict(), os.path.join(path_dir, 'model', save_file))
    logger.info(f"Saved PyTorch Model State to {save_file}")

    # save train_losses
    for key in train_losses.keys():
        train_losses[key] = np.array(train_losses[key])
    scio.savemat(os.path.join(path_dir, 'log', f"{log_file_name}.mat"),
                 train_losses)

    ## Test
    # logger.info('Test')
    # settransformer.eval()
    #
    # test_files = [f'test_data_M{dim_input}_1.mat']
    # for test_file in test_files:
    #     path = os.path.join(path_dir, test_file)
    #     # data = scio.loadmat(path)
    #     data = h5py.File(path)
    #     logger.info(f"testing data path: {path}")
    #
    #     solutionset = torch.from_numpy(np.transpose(data.get('Data'))).float()      # [dataset_num, data_num, M]
    #     hv = torch.from_numpy(np.transpose(data.get('HVval'))).float()              # [dataset_num, 1]
    #     hv = torch.reshape(hv, (hv.shape[0],1,1))                                   # [dataset_num, num_outputs, dim_output]
    #     dataloader = DataLoader(TensorDataset(solutionset, hv), batch_size=batch_size)
    #
    #     loss_fn = nn.MSELoss(reduction='mean')
    #
    #     size = solutionset.shape[0]
    #     num_batches = len(dataloader)
    #     test_losses = {'loss': []}
    #     with torch.no_grad():
    #         with tqdm.tqdm(total=int(size/batch_size), file=sys.stdout) as pbar_test:
    #             for batch, (X, y) in enumerate(dataloader):  # all data train once in 1 epoch.
    #                 input, output = X.to(device), y.to(device)  # [bs, 100, 3] [bs, 1, 1]
    #                 pred = settransformer.forward_allow_nan(input)  # [bs, num_outputs, dim_output]  [bs,1,1]
    #                 loss = loss_fn(torch.log(pred), torch.log(output))
    #                 test_losses['loss'].append(float(loss))
    #                 test_output_update = f'loss: {float(loss):>7f}'
    #                 pbar_test.update(1)
    #                 pbar_test.set_description(f"testing phase {batch+1} -> {test_output_update}")
    #
    #     test_losses['loss'] = np.mean(test_losses['loss'])
    #     logger.info(f"Test -> loss: {test_losses['loss']}")

        # save test_losses
        #TBD
