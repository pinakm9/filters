import model2
import upf_ex
import warnings
warnings.filterwarnings("ignore")

# create model to feed the filters
model = model2.model(size = 10)
# run experiments
upf_ex.experiment(model = model, particle_counts = [16, 64, 256, 1024, 4096], num_exprs = 20, alpha = 1.0, beta = 0.0, kappa = 2.0,\
                resampling_threshold = 0.1, titles = ['2D non-linear model'],\
                file_paths = ['../images/gsupf_results/model2_rmse_vs_num_particles.png'])
