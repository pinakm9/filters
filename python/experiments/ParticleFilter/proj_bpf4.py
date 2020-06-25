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
proj_model, d = model4.proj_model(size = int(sys.argv[1]))
model, d = model4.model(size = int(sys.argv[1]))
hidden_path = model.hidden_state.generate_path()
observed_path = model.observation.generate_path(hidden_path)
proj_op = model4.proj_obeserved_path(observed_path)
# solution using an implcit particle filter
bpf = fl.ParticleFilter(proj_model, particle_count = 2000)
bpf.update(proj_op, threshold_factor = 0.1, method = 'mean')

# plot trajectories
image_dir = str(script_path.parent.parent.parent) + '/images/ParticleFilter/'
model_name = 'proj_model4'
bpf.plot_trajectories(hidden_path, coords_to_plot = [0, 1, int(d/2), d-1], show = True, file_path = image_dir + model_name + '_trajectories.png')
bpf.compute_error(hidden_path)
bpf.plot_error(show = True, file_path = image_dir + model_name + '_abs_err_vs_time.png')
exact_final_mean = model4.update(observed_path[1:])[0][-1]
bpf_final_mean = bpf.computed_trajectory[-1]
print("Error in (final) mean = {}".format(np.linalg.norm(bpf_final_mean - exact_final_mean)))
print("RMSE = {}".format(bpf.rmse))
