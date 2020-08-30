# Takes size (or length of Markov chain or final time) of models as the command line argument
# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_path = Path(dirname(realpath(__file__)))
module_dir = str(script_path.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
# import remaining modules
import numpy as np
import filter as fl
import scipy
import utility as ut
import matplotlib.pyplot as plt
import plot
import model5
import attract as atr
"""
Generate paths
"""
s = int(sys.argv[1])
model, a, b = model5.model(size = s)
hidden_path = model.hidden_state.generate_path()
observed_path = model.observation.generate_path(hidden_path)
"""
Solution using an attractor particle filter
"""
db_path = str(script_path.parent.parent.parent) + '/data/henon_attractor.h5'
attractor_sampler = atr.AttractorSampler(db_path = db_path)
apf = fl.AttractorPF(model, particle_count = 500, attractor_sampler = attractor_sampler)
apf.update(observed_path, threshold_factor = 0.1, method = 'mean')

# plot trajectories
model = 'model5'
image_dir = str(script_path.parent.parent.parent) + '/images/AttractorPF/'
apf.plot_trajectories(hidden_path, coords_to_plot = [0, 1], show = True, file_path = image_dir + '{}_trajectories.png'.format(model))
apf.compute_error(hidden_path)
apf.plot_error(show = True, file_path = image_dir + '{}_abs_err_vs_time.png'.format(model), semilogy = True)
plt.figure(figsize = (8,8))
ax = plt.subplot(111)
ax.scatter(hidden_path[:, 0], hidden_path[:, 1], color = 'orange', s = 0.2)
plt.savefig(image_dir + '{}_true_trajectory.png'.format(model))
