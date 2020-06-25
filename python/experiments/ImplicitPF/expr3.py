# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_path = Path(dirname(realpath(__file__)))
module_dir = str(script_path.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
# import remaining modules
import model3
import experiment
import warnings
#warnings.filterwarnings("ignore")

# create model to feed the filters
model, a, b = model3.model(size = 10)
# run experiments
image_dir = str(script_path.parent.parent.parent) + '/images/ImplicitPF/'
model_name = 'Henon({}, {})'.format(a, b)
experiment.experiment(model = model, filter = 'ImplicitPF', particle_counts = [50, 100, 150, 200, 250], num_exprs = 20,\
                resampling_threshold = 0.1, titles = ['2D non-linear model'],\
                file_paths = [image_dir + model_name + '_rmse_vs_num_particles.png'],\
                final_exact_mean = None, F = model3.F, argmin_F = model3.argmin_F, grad_F = model3.grad_F)
