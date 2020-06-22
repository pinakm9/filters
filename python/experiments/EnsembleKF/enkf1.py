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
enkf = fl.EnsembleKF(model, ensemble_size = 2000)
enkf.update(observed_path)
# plot trajectories
image_dir = str(script_path.parent.parent.parent) + '/images/EnsembleKF/'
enkf.plot_trajectories(hidden_path, coords_to_plot = [0, 1], show = True, \
            file_path = image_dir + 'linear_trajectories.png')
enkf.compute_error(hidden_path)
enkf.plot_error(show = True, file_path = image_dir + 'linear_abs_err_vs_time.png')
enkf_final_mean = enkf.computed_trajectory[-1]
exact_final_mean = model1.update(observed_path[1:])[0][-1]
print("Error in (final) mean = {}".format(np.linalg.norm(enkf_final_mean - exact_final_mean)))
