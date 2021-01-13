"""
Plots evolution of ensembles
"""
# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
# import remaining modules
import filter as fl
import config as cf
import numpy as np
import Lorenz63_xy
from bpf_plotter import plot_ensemble_evol

# set random seed
seed = 2021
np.random.seed(seed)

# set model
ev_time = 50
prior_cov = 1.0
obs_cov = 0.1
shift = 0.0
obs_gap = 0.2
model, gen_path = Lorenz63_xy.get_model(size=ev_time, prior_cov=prior_cov, obs_cov=obs_cov,  shift=shift, obs_gap=obs_gap)

# set parameters
particle_count = 200
resampling_method = 'systematic_noisy'
resampling_threshold = 0.1
noise = 0.001

# set up configuration
config = {'Particle count': particle_count, 'Resampling method': resampling_method, 'Resampling threshold': resampling_threshold,\
          'Model': 'Lorenz63','Number of assimilation steps': ev_time, 'Observation gap': obs_gap, 'Numpy seed': seed, 'Deterministic': True,\
          'Prior covariance': prior_cov, 'Observation covariance': obs_cov, 'Resampling covariance': noise, 'Shift': shift * np.ones(3)}

expr_name = 'xy_np_{}_rn_{}_pc_{}_oc_{}_og_{}_s_{}'.format(particle_count, noise, prior_cov, obs_cov, obs_gap, shift)
cc = cf.ConfigCollector(expr_name = expr_name, folder = str(script_dir))
cc.add_params(config)
cc.write()

# assimilation using a bootstrap particle filter
hidden_path = gen_path(ev_time)
np.save(cc.res_path + '/hidden_path.npy', hidden_path)
observed_path = model.observation.generate_path(hidden_path)

bpf = fl.ParticleFilter(model, particle_count = particle_count, record_path = cc.res_path + '/bpf_assimilation.h5')
bpf.update(observed_path, resampling_method = resampling_method, threshold_factor = resampling_threshold, method = 'mean', noise=noise)
plot_ensemble_evol(cc.res_path + '/bpf_assimilation.h5', hidden_path, time_factor=1, pt_size=80, obs_inv=True)
