# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
module_dir = str(Path(dirname(realpath(__file__))).parent.parent)
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
experiment.experiment(model = model, filter = 'GlobalSamplingUPF', particle_counts = [16, 64, 256, 1024, 4096], num_exprs = 20,\
                resampling_threshold = 0.1, titles = ['2D linear model']*2,\
                file_paths = ['../images/gsupf_results/linear_rmse_vs_num_particles.png', '../images/gsupf_results/linear_err_in_mean_vs_num_particles.png'],\
                final_exact_mean = final_exact_mean, alpha = 1.0, beta = 0.0, kappa = 2.0)
