import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
import torch

def calc_convergence_fea(candidate_temp, ref):
    candidate = candidate_temp[~np.isnan(candidate_temp).any(axis=1)]
    feature_temp = ref - candidate
    feature_add = np.abs(np.prod(feature_temp, axis = 1)).reshape(candidate.shape[0],1)
    feature = np.concatenate((feature_temp, feature_add), axis=1)
    return feature


def calc_diveristy_feature(solution, selected_set):
    distances1 = cdist(solution, selected_set, metric='cosine')
    min_cosine_sim = np.min(distances1, axis=1)
    
    distances2 = cdist(solution, selected_set, metric='cosine')
    max_cosine_sim = np.max(distances2, axis=1)
    
    distances3 = cdist(solution, selected_set, metric='cosine')
    mean_cosine_sim = np.mean(distances3, axis=1)
    
    distances4 = cdist(solution, selected_set, metric='euclidean')
    min_cosine_dis = np.min(distances4, axis=1)
    
    distances5 = cdist(solution, selected_set, metric='euclidean')
    max_cosine_dis = np.max(distances5, axis=1)
    
    distances6 = cdist(solution, selected_set, metric='euclidean')
    mean_cosine_dis = np.mean(distances6, axis=1)
    
    return min_cosine_sim, max_cosine_sim, mean_cosine_sim, min_cosine_dis, max_cosine_dis, mean_cosine_dis