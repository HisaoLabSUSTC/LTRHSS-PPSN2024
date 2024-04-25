import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
import torch.nn.functional as F
from feature import calc_convergence_fea, calc_diveristy_feature
import numpy as np

class LTRHSS(nn.Module):
    def __init__(self, device, obj_num):
        super(LTRHSS, self).__init__()
        
        self.num_objective = obj_num
        # self.selected_num = args.selected_num
        self.device = device
        
        self.w1 = nn.Embedding(1, self.num_objective + 1).to(self.device)
        self.w2 = nn.Embedding(1, 6).to(self.device)
        
        self.w1.weight.data = torch.nn.init.normal_(self.w1.weight.data, 0, 0.01)
        self.w2.weight.data = torch.nn.init.normal_(self.w2.weight.data, 0, 0.01)
        
        self.myparameters = [self.w1, self.w2]
        
    def forward(self, candidate_temp, rank_temp, feature):
        candidate = candidate_temp[~np.isnan(candidate_temp).any(axis=1)]
        y = rank_temp[~np.isnan(rank_temp)].astype(int) - 1
        selected_solution_num = len(y)
        feature = torch.from_numpy(feature).to(self.device)
        # w1 = self.w1.weight.data
        # w2 = self.w2.weight.data
        w1 = self.w1.weight
        w2 = self.w2.weight
        selected_solution_index = []
        solution_1 = candidate[y[0], :]
        solution_1_conver_fea = feature[y[0],:]
        selected_solution_index.append(y[0])
        # aa = torch.sum(torch.mul(w1, solution_1_conver_fea), dim=1)
        # bb = torch.exp(torch.sum(torch.mul(w1, solution_1_conver_fea), dim=1))
        # cc = torch.sum(torch.mul(w1, feature), dim=1)
        # dd = torch.exp(torch.sum(torch.mul(w1, feature), dim=1))
        # ee = torch.sum(torch.exp(torch.sum(torch.mul(w1, feature), dim=1)))
        
        loss = -1 * torch.log(torch.exp(torch.sum(torch.mul(w1, solution_1_conver_fea), dim=1)) / torch.sum(torch.exp(torch.sum(torch.mul(w1, feature), dim=1))))
        y_temp = y[1:selected_solution_num]
        for yy in y_temp:
            current_solution = candidate[yy,:].reshape(1, self.num_objective)
            selected_solutions = candidate[selected_solution_index,:]
            remain_solutions = candidate[~np.isin(np.arange(candidate.shape[0]), selected_solution_index),:]
                       
            solution_convergence = torch.sum(torch.mul(w1, feature[yy,:]), dim=1)           
            div_fea1, div_fea2, div_fea3, div_fea4, div_fea5, div_fea6 = calc_diveristy_feature(current_solution, selected_solutions)
            diveristy_fea = torch.from_numpy(np.vstack((div_fea1, div_fea2, div_fea3, div_fea4, div_fea5, div_fea6)).T).to(self.device)
            solution_diversity = torch.sum(torch.mul(w2, diveristy_fea), dim=1)
            solution_prob = torch.exp(solution_convergence + solution_diversity)
            
            set_feature = feature[~np.isin(np.arange(candidate.shape[0]), selected_solution_index),:]
            set_convergence =  torch.sum(torch.mul(w1, set_feature), dim=1)
            div_set_fea1, div_set_fea2, div_set_fea3, div_set_fea4, div_set_fea5, div_set_fea6 = calc_diveristy_feature(remain_solutions, selected_solutions)
            div_set_fea = torch.from_numpy(np.vstack((div_set_fea1, div_set_fea2, div_set_fea3, div_set_fea4, div_set_fea5, div_set_fea6)).T).to(self.device)           
            set_diversity = torch.sum(torch.mul(w2, div_set_fea), dim=1)
            # set_diversity1 = torch.exp(torch.sum(torch.mul(w2, div_set_fea), dim=1))
            set_score = torch.exp(set_convergence + set_diversity)
            set_prob = torch.sum(set_score)
            
            selected_solution_index.append(yy)
            
            current_loss = -1 * torch.log(solution_prob / set_prob)
            loss = loss + current_loss
            
        return loss
    
    def eval_LTR(self, candidate_temp, feature):
        candidate = candidate_temp[~np.isnan(candidate_temp).any(axis=1)]
        feature = torch.from_numpy(feature).to(self.device)
        w1 = self.w1.weight
        w2 = self.w2.weight
        y_pred = []
        y_pred_1 = torch.argmax(torch.sum(torch.mul(w1, feature), dim=1))
        y_pred.append(y_pred_1.item())
        selected_num = int(candidate.shape[0])
        for i in range(selected_num-1):
            selected_solutions = candidate[y_pred,:]
            # remained_solutions = candidate[~np.isin(np.arange(candidate.shape[0]), y_pred),:]
            # set_feature = feature[~np.isin(np.arange(candidate.shape[0]), y_pred),:]
            set_convergence = torch.sum(torch.mul(w1, feature), dim=1)
            div_set_fea1, div_set_fea2, div_set_fea3, div_set_fea4, div_set_fea5, div_set_fea6 = calc_diveristy_feature(candidate, selected_solutions)
            div_set_fea = torch.from_numpy(np.vstack((div_set_fea1, div_set_fea2, div_set_fea3, div_set_fea4, div_set_fea5, div_set_fea6)).T).to(self.device)
            set_diversity = torch.exp(torch.sum(torch.mul(w2, div_set_fea), dim=1))
            set_score = set_convergence + set_diversity
            set_score[y_pred] = -np.inf
            y_pred_temp = torch.argmax(set_score)
            y_pred.append(y_pred_temp.item())
            # y_pred = np.array(y_pred)
        return y_pred
            
            
        
        
            
            
            
            
            
            
            
            
            
            
        
         
        
        
        