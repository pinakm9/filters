# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_path = Path(dirname(realpath(__file__)))
module_dir = str(script_path.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
# import remaining modules
import model1
import experiment
import warnings
warnings.filterwarnings("ignore")

# create model to feed the filters
model = model1.model(size = 10)
final_exact_mean = lambda observed_path: model1.update(observed_path)[0][-1]
# run experiments
image_dir = str(script_path.parent.parent.parent) + '/images/ImplicitPF/'
model_name = 'model1'
experiment.experiment(model = model, filter = 'ImplicitPF', particle_counts = [25, 50, 100, 200, 400], num_exprs = 30,\
                resampling_threshold = 0.1, titles = ['2D linear model']*2,\
                file_paths = [image_dir + model_name + '_rmse_vs_num_particles.png', image_dir + model_name + '_err_in_mean_vs_num_particles.png'],\
                final_exact_mean = final_exact_mean, F = model1.F, argmin_F = model1.argmin_F, grad_F = model1.grad_F)
