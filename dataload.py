from scipy.io import loadmat
import numpy as np
from feature import calc_convergence_fea

pf_shape = 'Concave'
candidate_size = 200
num_objective = 3
run = 1

data_name = '{}_N{}_M{}_{}.mat'.format(pf_shape, candidate_size, num_objective, run)
mat_data = loadmat('./data/candidate/{}'.format(data_name))
candidates = mat_data['uu'].T


reference_point = np.ones((1, num_objective))*1.1

convergence_fea = calc_convergence_fea(candidates, reference_point)

list_rank = np.arange(100)


print('end')