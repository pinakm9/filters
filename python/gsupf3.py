import numpy as np
import filter as fl
import scipy
import utility as ut
import matplotlib.pyplot as plt
import plot
import model3
np.random.seed(16)

"""
Solution using a gsupf
"""
model, a, b = model3.model()
hidden_path = model.hidden_state.generate_path()
observed_path = model.observation.generate_path(hidden_path)
#"""
plot.SignalPlotter(signals = [hidden_path, observed_path]).plot_signals(labels = ['hidden', 'observed'],\
    styles = [{'linestyle': 'solid'}, {'marker': 'x'}], colors = ['black', 'blue'],  coords_to_plot = [0, 1], show = True)
#"""
pf = fl.GlobalSamplingUPF(model, particle_count = 4000, alpha = 1.0, kappa = 2.0, beta = 0.0)
pf.update(observed_path, threshold_factor = 0.1, method = 'mean')
mean = pf.computed_trajectory[-1]

"""
plt.scatter(*zip(*pf.particles))
plt.scatter([mu[0]], [mu[1]], color = 'red')
plt.scatter([mean[0]], [mean[1]], color = 'green')
plt.show()
"""

pf.plot_trajectories(hidden_path, coords_to_plot = [0, 1], show = True)
pf.compute_error(hidden_path)
pf.plot_error(show = True)
#print("\n\n error in mean {}\n\n".format(np.linalg.norm(mean - mu)))
