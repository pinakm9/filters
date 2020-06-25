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
import model1
"""
Generate paths
"""
s = int(sys.argv[1])
model = model1.model(size = s)
hidden_path = model.hidden_state.generate_path()
observed_path = model.observation.generate_path(hidden_path)
"""
Solution using an implcit particle filter
"""

ipf = fl.ImplicitPF(model, particle_count = 200, F = model1.F, argmin_F= model1.argmin_F, grad_F = model1.grad_F)
ipf.update(observed_path, threshold_factor = 0.1, method = 'mean')

# plot true vs computed mean
image_dir = str(script_path.parent.parent.parent) + '/images/ImplicitPF/'
model_name = 'model1'
exact_final_mean = model1.update(observed_path[1:])[0][-1]
ipf_final_mean = ipf.computed_trajectory[-1]
plt.scatter(*zip(*ipf.particles))
plt.scatter([exact_final_mean[0]], [exact_final_mean[1]], label = 'True mean', color = 'red')
plt.scatter([ipf_final_mean[0]], [ipf_final_mean[1]], label = 'Filter mean', color = 'green')
plt.title('Mean at time = {}, #particles = {}'.format(s-1, ipf.particle_count))
plt.legend()
plt.savefig(image_dir + model_name + '_final_mean.png')
plt.show()

# plot trajectories
ipf.plot_trajectories(hidden_path, coords_to_plot = [0, 1], show = True, file_path = image_dir + model_name + '_trajectories.png')
ipf.compute_error(hidden_path)
ipf.plot_error(show = True, file_path = image_dir + model_name + '_abs_err_vs_time.png')
print("Error in (final) mean = {}".format(np.linalg.norm(ipf_final_mean - exact_final_mean)))
