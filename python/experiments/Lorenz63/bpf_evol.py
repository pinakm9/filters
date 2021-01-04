"""
Plots evolution of ensembles
"""
# Takes size (or length of Markov chain or final time) of models as the command line argument
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
import Lorenz63
from ens_plot import plot_ensemble_evol

# set model
ev_time = 50
model = Lorenz63.model(size=50)

# set parameters
particle_count = 5000
resampling_threshold = 0.1


# set up configuration
config = {'Particle count': particle_count, 'Resampling threshold': resampling_threshold, 'Model': 'Lorenz63'}
cc = cf.ConfigCollector(expr_name = 'Evolution', folder = str(script_dir))
cc.add_params(config)
cc.write()

# assimilation using a bootstrap particle filter
hp = Lorenz63.gen_path(length = ev_time)
hidden_path = hp#np.zeros((50, 3))
"""
for i, e in enumerate(hp):
    if i%10 == 0:
        hidden_path[int(i/10)] = e
"""
observed_path = model.observation.generate_path(hidden_path)
#print(hidden_path)
#print(observed_path)
bpf = fl.ParticleFilter(model, particle_count = particle_count, record_path = cc.res_path + '/bpf_assimilation.h5')
bpf.update(observed_path, threshold_factor = resampling_threshold, method = 'mean')
bpf.plot_ensembles(cc.res_path + '/bpf_assimilation.h5', hidden_path, obs_inv = True, pt_size = 80)
