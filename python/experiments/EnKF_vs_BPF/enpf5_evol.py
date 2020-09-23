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
import numpy as np
import filter as fl
import attract as atr
import model5
import config as cf

# set parameters
ev_time = int(sys.argv[1])
model, a, b, x0 = model5.model(size = ev_time)
particle_count = 100
resampling_threshold = 0.1
num_exprs = 10

# set up configuration
config = {'Henon a': a, 'Henon b': b, 'Particle count': particle_count, 'Number of experiments': num_exprs, 'Resampling threshold': resampling_threshold,\
          'Evolution time': ev_time, 'Model': 'model5', 'Starting point': x0, 'Starting pt randomly generated on attractor': 'No, in large db'}
cc = cf.ConfigCollector(expr_name = 'Evolution', folder = script_dir)
cc.add_params(config)
cc.write()

# assimilation using a bootstrap particle filter
hidden_path = model5.gen_path(length = ev_time)
observed_path = model.observation.generate_path(hidden_path)
bpf = fl.ParticleFilter(model, particle_count = particle_count, record_path = cc.res_path + '/bpf_assimilation.h5')
bpf.update(observed_path, threshold_factor = resampling_threshold, method = 'mean')
bpf.plot_ensembles(hidden_path, obs_inv = True)
