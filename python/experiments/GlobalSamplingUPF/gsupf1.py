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
import model1
"""
Generate paths
"""
s = int(sys.argv[1])
model = model1.model(size = s)
hidden_path = model.hidden_state.generate_path()
observed_path = model.observation.generate_path(hidden_path)
"""
Solution using a GS-UPF
"""
means, covs = model1.update(observed_path[1:])
mu = means[-1] # true final mean
pf = fl.GlobalSamplingUPF(model, particle_count = 2000, alpha = 1, kappa = 2, beta = 2)
pf.update(observed_path, threshold_factor = 0.1, method = 'mean')
mean = pf.computed_trajectory[-1]

# plot true vs computed mean
plt.scatter(*zip(*pf.particles))
plt.scatter([mu[0]], [mu[1]], label = 'True mean', color = 'red')
plt.scatter([mean[0]], [mean[1]], label = 'Filter mean', color = 'green')
plt.title('Mean at time = {}, #particles = {}'.format(s-1, pf.particle_count))
plt.legend()
plt.savefig('../images/gsupf_results/linear_final_mean.png')
plt.show()

# plot trajectories
pf.plot_trajectories(hidden_path, coords_to_plot = [0, 1], show = True, \
            file_path = '../images/gsupf_results/linear_trajectories.png')
pf.compute_error(hidden_path)
pf.plot_error(show = True, file_path = '../images/gsupf_results/linear_abs_err_vs_time.png')
print("Error in (final) mean = {}".format(np.linalg.norm(mean - mu)))
