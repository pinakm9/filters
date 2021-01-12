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
import Lorenz63_ia as Lorenz63
from bpf_plotter import plot_ensemble_evol

# set model
ev_time = 50
model, prior_cov, obs_cov = Lorenz63.model(size=ev_time)

# set parameters
particle_count = 800
resampling_method = 'systematic_noisy'
resampling_threshold = 0.1
seed = 2021
noise = 0.05
np.random.seed(seed)

# set up configuration
config = {'Particle count': particle_count, 'Resampling method': resampling_method, 'Resampling threshold': resampling_threshold,\
          'Model': 'Lorenz63 inaccurate','Number of assimilation steps': ev_time, 'Observation gap': 0.2, 'Numpy seed': seed, 'Deterministic':\
          True, 'Prior covariance': prior_cov, 'Observation covariance': obs_cov, 'Resampling covariance': noise, 'shift': 3.0}
cc = cf.ConfigCollector(expr_name = 'Evolution', folder = str(script_dir))
cc.add_params(config)
cc.write()

# assimilation using a bootstrap particle filter
hidden_path = Lorenz63.gen_path(length = ev_time)
np.save(cc.res_path + '/hidden_path.npy', hidden_path)
observed_path = model.observation.generate_path(hidden_path)

bpf = fl.ParticleFilter(model, particle_count = particle_count, record_path = cc.res_path + '/bpf_assimilation.h5')
bpf.update(observed_path, resampling_method = resampling_method, threshold_factor = resampling_threshold, method = 'mean', noise=noise)
plot_ensemble_evol(cc.res_path + '/bpf_assimilation.h5', hidden_path, time_factor=1, pt_size=80, obs_inv=True)
