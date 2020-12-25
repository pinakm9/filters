# Takes size (or length of Markov chain or final time) of models as the command line argument
# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import tables
import numpy as np
import ens_plot as enp

root_data_folder = 'ob1-20201225T132638Z-001/ob1'
observation_file = 'obs1_gap_0.2_H1__mu=0.1_obs_cov1.npy'
prior_file = 'obs=3_ens=50_Mcov=5,ocov=0.1_,gap=0.2_alpha=1.0_loc=convex_r=0f_ensemble.npy'
posterior_file = 'obs=3_ens=50_Mcov=5,ocov=0.1_,gap=0.2_alpha=1.0_loc=convex_r=0a_ensemble.npy'

def shorten(arr, factor=10):
    arr_shape = list(arr.shape)
    arr_shape[0] = int(arr_shape[0]/10)
    new_arr = np.zeros(arr_shape)
    for i, elem in enumerate(arr):
        if i%10 == 0:
            new_arr[int(i/10)] = elem
    return new_arr

def make_db(dim, db_path, observation_path, prior_ensemble_path, posterior_ensemble_path, factor=10):
    hdf5 = tables.open_file(db_path, 'a')
    # set point description
    point_description = {}
    for i in range(dim):
        point_description['x' + str(i)] = tables.Float64Col(pos = i)

    # create necessary files and folders
    file = hdf5.create_table('/', 'observation', point_description)
    file.flush()
    hdf5.create_group('/', 'prior_ensemble')
    hdf5.create_group('/', 'posterior_ensemble')
    # write observations
    file = hdf5.root.observation
    file.append(shorten(np.load(observation_path), factor))
    file.flush()

    # write prior_ensembles
    folder = hdf5.root.prior_ensemble
    prior_ensemble = np.load(prior_ensemble_path)
    for i in range(0, prior_ensemble.shape[0], factor):
        file = hdf5.create_table(hdf5.root.prior_ensemble, 'time_{}'.format(i), point_description)
        file.append(prior_ensemble[i, :, :].T)
        file.flush()

    # write posterior_ensembles
    folder = hdf5.root.posterior_ensemble
    posterior_ensemble = np.load(posterior_ensemble_path)
    for i in range(0, posterior_ensemble.shape[0], factor):
        file = hdf5.create_table(hdf5.root.posterior_ensemble, 'time_{}'.format(i), point_description)
        file.append(posterior_ensemble[i, :, :].T)
        file.flush()

    return hdf5

    """
    point_description
    # write down observations
    observation = np.load(observation_path)
    hdf5.root.observation.append
    ensemble.flush()
    """
db_path = 'enkf_assimilation.h5'
observation_path = root_data_folder + '/' + observation_file
prior_ensemble_path = root_data_folder + '/' + prior_file
posterior_ensemble_path = root_data_folder + '/' + posterior_file

#hdf5 = make_db(3, db_path, observation_path, prior_ensemble_path, posterior_ensemble_path)
#hdf5.close()

hidden_path = shorten(np.load('Trajectory_0.2_.npy'))
enp.plot_ensemble_evol(db_path, hidden_path, time_factor=10, pt_size=80, obs_inv=True)
