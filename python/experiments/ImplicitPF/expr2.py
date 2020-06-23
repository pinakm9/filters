# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_path = Path(dirname(realpath(__file__)))
module_dir = str(script_path.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
# import remaining modules
import model2
import experiment
import warnings
#warnings.filterwarnings("ignore")

# create model to feed the filters
model = model2.model(size = 10)
# run experiments
image_dir = str(script_path.parent.parent.parent) + '/images/ImplicitPF/'
model_name = 'model2'
experiment.experiment(model = model, filter = 'ImplicitPF', particle_counts = [ 8, 32, 128], num_exprs = 20,\
                resampling_threshold = 0.1, titles = ['1D non-linear model']*2,\
                file_paths = [image_dir + model_name + '_rmse_vs_num_particles.png', image_dir + model_name + '_err_in_mean_vs_num_particles.png'],\
                final_exact_mean = None, F = model2.F, argmin_F = model2.argmin_F, grad_F = model2.grad_F)
