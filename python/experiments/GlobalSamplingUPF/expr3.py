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
final_exact_mean = lambda observed_path: model3.update(observed_path)[0][-1]
# run experiments
image_dir = str(script_path.parent.parent.parent) + '/images/GlobalSamplingUPF/'
model_name = 'model3'
experiment.experiment(model = model, filter = 'GlobalSamplingUPF', particle_counts = [20, 40, 80, 320], num_exprs = 20,\
                resampling_threshold = 0.1, titles = ['2D non-linear model'],\
                file_paths = [image_dir + model_name + '_rmse_vs_num_particles.png'],\
                final_exact_mean = final_exact_mean, alpha = 1.0, kappa = 2.0, beta = 0.0)
