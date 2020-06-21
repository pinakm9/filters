# Takes size (or length of Markov chain or final time) of models as the command line argument
import numpy as np
import filter as fl
import scipy
import utility as ut
import matplotlib.pyplot as plt
import plot
import model2
import sys
np.random.seed(16)
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
pf = fl.GlobalSamplingUPF(model, particle_count = 2000, alpha = 1.0, kappa = 2.0, beta = 0.0)
pf.update(observed_path, threshold_factor = 0.1, method = 'mean')

# plot trajectories
pf.plot_trajectories(hidden_path, coords_to_plot = [0], show = True, \
            file_path = '../images/gsupf_results/model2_trajectories.png')
pf.compute_error(hidden_path)
pf.plot_error(show = True, file_path = '../images/gsupf_results/model2_abs_err_vs_time.png')
