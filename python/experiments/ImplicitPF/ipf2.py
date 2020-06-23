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
import model2
#np.random.seed(16)
"""
Generate paths
"""
s = int(sys.argv[1])
model = model2.model(size = s)
hidden_path = model.hidden_state.generate_path()
observed_path = model.observation.generate_path(hidden_path)
#"""
plot.SignalPlotter(signals = [hidden_path, observed_path]).plot_signals(labels = ['hidden', 'observed'],\
    styles = [{'linestyle': 'solid'}, {'marker': 'x'}], colors = ['black', 'blue'],  coords_to_plot = [0], show = True)
#"""
"""
Solution using a GS-UPF
"""
ipf = fl.ImplicitPF(model, particle_count = 200, F = model2.F, argmin_F = model2.argmin_F, grad_F = model2.grad_F)
ipf.update(observed_path, threshold_factor = 0.1, method = 'mean')

# plot trajectories
image_dir = str(script_path.parent.parent.parent) + '/images/ImplicitPF/'
model_name = 'model2'
ipf.plot_trajectories(hidden_path, coords_to_plot = [0], show = True, file_path = image_dir + model_name + '_trajectories.png')
ipf.compute_error(hidden_path)
ipf.plot_error(show = True, file_path = image_dir + model_name + '_abs_err_vs_time.png')
