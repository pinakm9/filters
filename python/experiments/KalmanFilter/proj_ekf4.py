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
"""
Generate paths
"""
s = int(sys.argv[1])
model, d = model4.proj_model(size = s)
hidden_path = model.hidden_state.generate_path()
observed_path = model.observation.generate_path(hidden_path)
"""
Solution using an implcit particle filter
"""
ekf = fl.KalmanFilter(model, mean0 = np.zeros(d), cov0 = np.identity(d))
ekf.update(observed_path)
# plot trajectories
image_dir = str(script_path.parent.parent.parent) + '/images/KalmanFilter/'
model_name = 'proj_model4'
ekf.plot_trajectories(hidden_path, coords_to_plot = [0, 1], show = True, \
            file_path = image_dir + model_name + '_trajectories.png')
ekf.compute_error(hidden_path)
ekf.plot_error(show = True, file_path = image_dir +  model_name + '_abs_err_vs_time.png')
ekf_final_mean = ekf.computed_trajectory[-1]
exact_final_mean = model4.proj_update(observed_path[1:])[0][-1]
print("Error in (final) mean = {}\nRMSE = {}".format(np.linalg.norm(ekf_final_mean - exact_final_mean), ekf.rmse))
