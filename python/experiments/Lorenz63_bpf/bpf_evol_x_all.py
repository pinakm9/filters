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
import Lorenz63_x
from bpf_plotter import plot_ensemble_evol


experiments = [0] * 5
experiments[0] = {'np': 100, 'rn': 0.05, 'pc': 1.0, 'oc': 0.1, 'og': 0.2, 's': 0.0}
experiments[1] = {'np': 200, 'rn': 0.01, 'pc': 1.0, 'oc': 0.1, 'og': 0.2, 's': 0.0}
experiments[2] = {'np': 100, 'rn': 0.05, 'pc': 0.01, 'oc': 0.1, 'og': 0.2, 's': 2.0}
experiments[3] = {'np': 800, 'rn': 0.05, 'pc': 0.01, 'oc': 0.1, 'og': 0.2, 's': 3.0}
experiments[4] = {'np': 100, 'rn': 0.05, 'pc': 0.1, 'oc': 0.1, 'og': 0.2, 's': 3.0}

# set random seed
seed = 2021
np.random.seed(seed)

for i, e in enumerate(experiments):
    print('commencing experiment #{} ...'.format(i))
    # set model
    ev_time = 50
    prior_cov = e['pc']
    obs_cov = e['oc']
    shift = e['s']
    obs_gap = e['og']
    model, gen_path = Lorenz63_x.get_model(size=ev_time, prior_cov=prior_cov, obs_cov=obs_cov,  shift=shift, obs_gap=obs_gap)

    # set parameters
    particle_count = e['np']
    resampling_method = 'systematic_noisy'
    resampling_threshold = 0.1
    noise = e['rn']

    # set up configuration
    config = {'Particle count': particle_count, 'Resampling method': resampling_method,
              'Resampling threshold': resampling_threshold, 'Model': 'Lorenz63','Number of assimilation steps': ev_time,\
              'Observation gap': obs_gap, 'Numpy seed': seed, 'Deterministic': True,\
              'Prior covariance': prior_cov, 'Observation covariance': obs_cov, 'Resampling covariance': noise,\
              'Shift': shift * np.ones(3)}
    expr_name = 'xy_np_{}_rn_{}_pc_{}_oc_{}_og_{}_s_{}'.format(particle_count, noise, prior_cov, obs_cov, obs_gap, shift)
    cc = cf.ConfigCollector(expr_name = expr_name, folder = str(script_dir))
    cc.add_params(config)
    cc.write()

    # assimilation using a bootstrap particle filter
    hidden_path = gen_path(ev_time)
    np.save(cc.res_path + '/hidden_path.npy', hidden_path)
    observed_path = model.observation.generate_path(hidden_path)

    bpf = fl.ParticleFilter(model, particle_count = particle_count, record_path = cc.res_path + '/bpf_assimilation.h5')
    bpf.update(observed_path, resampling_method=resampling_method, threshold_factor=resampling_threshold, method='mean',\
               noise=noise)
    plot_ensemble_evol(cc.res_path + '/bpf_assimilation.h5', hidden_path, time_factor=1, pt_size=80, obs_inv=True)
