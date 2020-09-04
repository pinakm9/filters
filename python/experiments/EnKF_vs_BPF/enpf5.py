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
"""
Generate paths
"""
s = int(sys.argv[1])
model, a, b = model5.model(size = s)
hidden_path = model5.gen_path(length = s)
observed_path = model.observation.generate_path(hidden_path)
"""
Solution using an implcit particle filter
"""
ensemble_size = 100
ensemble_trajectory = np.zeros((s, 2, ensemble_size))
enkf = fl.EnsembleKF(model, ensemble_size = ensemble_size)
for i, observation in enumerate(observed_path):
    enkf.update([observation])
    ensemble_trajectory[i] = enkf.ensemble
"""
Solution using a particle filter
"""
particle_count = 100
resampling_threshold = 0.1
particles_trajectory = np.zeros((s, 2, particle_count))
bpf = fl.ParticleFilter(model, particle_count = particle_count)
for i, observation in enumerate(observed_path):
    bpf.update([observation], threshold_factor = resampling_threshold, method = 'mean')
    particles_trajectory[i] = bpf.particles.T
"""
plot trajectories
"""
image_dir = str(script_path.parent.parent.parent) + '/images/EnKF_vs_BPF/'
plot.SignalPlotter(signals = [enkf.computed_trajectory, bpf.computed_trajectory]).plot_signals(labels = ['EnKF', 'BPF'], styles = [{'linestyle':'solid'}, {'marker':'o'}],\
                plt_fns = ['plot', 'scatter'], colors = ['red', 'blue'], coords_to_plot = [0, 1])
"""
plot ensembles
"""
ax = plot.plot_ensemble_trajectories([particles_trajectory], colors = ['red', 'blue'], show = False)
ax.scatter(hidden_path[:, 0], hidden_path[:, 1], color = 'black')
plt.savefig(image_dir + 'particles.png')
ax = plot.plot_ensemble_trajectories([ensemble_trajectory], colors = ['red', 'blue'], show = False)
ax.scatter(hidden_path[:, 0], hidden_path[:, 1], color = 'black')
plt.savefig(image_dir + 'ensemble.png')
#plt.show()
