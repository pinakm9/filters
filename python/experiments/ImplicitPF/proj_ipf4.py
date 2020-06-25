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
import model4

# generate paths
model, d = model4.proj_model(size = int(sys.argv[1]))
hidden_path = model.hidden_state.generate_path()
observed_path = model.observation.generate_path(hidden_path)

# solution using an implcit particle filter
ipf = fl.ImplicitPF(model, particle_count = 500, F = model4.proj_F, argmin_F= model4.proj_argmin_F, grad_F = model4.proj_grad_F)
ipf.update(observed_path, threshold_factor = 0.1, method = 'mean')

# plot trajectories
image_dir = str(script_path.parent.parent.parent) + '/images/ImplicitPF/'
model_name = 'proj_model4'
ipf.plot_trajectories(hidden_path, coords_to_plot = [0, 1, int(d/2), d-1], show = True, file_path = image_dir + model_name + '_trajectories.png')
ipf.compute_error(hidden_path)
ipf.plot_error(show = True, file_path = image_dir + model_name + '_abs_err_vs_time.png')
exact_final_mean = model4.proj_update(observed_path[1:])[0][-1]
ipf_final_mean = ipf.computed_trajectory[-1]
print("Error in (final) mean = {}".format(np.linalg.norm(ipf_final_mean - exact_final_mean)))
print("RMSE = {}".format(ipf.rmse))
