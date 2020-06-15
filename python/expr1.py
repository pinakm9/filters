import model1
import upf_ex
import warnings
warnings.filterwarnings("ignore")

# create model to feed the filters
model = model1.model(size = 10)
final_exact_mean = lambda observed_path: model1.update(observed_path)[0][-1]
# run experiments
upf_ex.experiment(model = model, particle_counts = [16, 64, 256, 1024, 4096], num_exprs = 20, alpha = 1.0, beta = 0.0, kappa = 2.0,\
                resampling_threshold = 0.1, titles = ['2D linear model']*2,\
                file_paths = ['../images/gsupf_results/linear_rmse_vs_num_particles.png', '../images/gsupf_results/linear_err_in_mean_vs_num_particles.png'],\
                final_exact_mean = final_exact_mean)
