# Takes size (or length of Markov chain or final time) of models as the command line argument
# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
module_dir = str(Path(dirname(realpath(__file__))).parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
# import remaining modules
import numpy as np
import filter as fl
import scipy
import utility as ut
import matplotlib.pyplot as plt
import plot
import model3
np.random.seed(16)

"""
Generate paths
"""
s = int(sys.argv[1])
model, a, b = model3.model(size = s)
hidden_path = model.hidden_state.generate_path()
observed_path = model.observation.generate_path(hidden_path)
#"""
plot.SignalPlotter(signals = [hidden_path, observed_path]).plot_signals(labels = ['hidden', 'observed'],\
    styles = [{'linestyle': 'solid'}, {'marker': 'x'}], colors = ['black', 'blue'],  coords_to_plot = [0, 1], show = True)
#"""
"""
Solution using a GS-UPF
"""
pf = fl.GlobalSamplingUPF(model, particle_count = 1000, alpha = 1.0, kappa = 2.0, beta = 0.0)
pf.update(observed_path, threshold_factor = 0.1, method = 'mean')

# plot trajectories
pf.plot_trajectories(hidden_path, coords_to_plot = [0, 1], show = True, \
            file_path = '../images/gsupf_results/Henon({}, {})_trajectories.png'.format(a, b))
pf.compute_error(hidden_path)
pf.plot_error(show = True, file_path = '../images/gsupf_results/Henon({}, {})_abs_err_vs_time.png'.format(a, b))
