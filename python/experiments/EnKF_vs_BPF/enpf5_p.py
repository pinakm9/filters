# Takes size (or length of Markov chain or final time) of models as the command line argument
# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
# import remaining modules
import numpy as np
import filter as fl
import scipy
import utility as ut
import matplotlib.pyplot as plt
import plot
import attract as atr
import model5
import config as cf
# set parameters
s = int(sys.argv[1])
model, a, b = model5.model(size = s)
db_id = '14_3_medium'
db_path = str(script_dir.parent.parent.parent) + '/data/henon_attractor_{}.h5'.format(db_id)
attractor_sampler = atr.AttractorSampler(db_path = db_path)
resample_id = '0'
particle_count = 100
resampling_threshold = 0.1
num_exprs = 10

# set up configuration
config = {'Henon a': a, 'Henon b': b, 'Attractor database': db_path, 'Attractor id': db_id, 'Attractor resampling scheme': resample_id,\
          'Particle count': particle_count, 'Number of experiments': num_exprs, 'Resampling threshold': resampling_threshold, 'Evolution time': s, 'Model': 'model5',
          'Starting point': 'randomly generated in the attractor'}
cc = cf.ConfigCollector(expr_name = 'EnKF vs BPF - performance', folder = script_dir)
cc.add_params(config)
cc.write()

# initialize error containers
enkf_abs_err = np.zeros(s)
bpf_abs_err = np.zeros(s)
apf_abs_err = np.zeros(s)

# generate paths and assimilate
for i in range(num_exprs):
    print('Working on experiment #{} ...'.format(i))
    # initialize filters
    enkf = fl.EnsembleKF(model, ensemble_size = particle_count)
    bpf = fl.ParticleFilter(model, particle_count = particle_count)
    apf = fl.AttractorPF(model, particle_count = particle_count, attractor_sampler = attractor_sampler)
    hidden_path = model5.gen_path(length = s)
    observed_path = model.observation.generate_path(hidden_path)
    enkf.update(observed_path)
    enkf.compute_error(hidden_path)
    enkf_abs_err += enkf.abs_error
    bpf.update(observed_path, threshold_factor = resampling_threshold, method = 'mean')
    bpf.compute_error(hidden_path)
    bpf_abs_err += bpf.abs_error
    apf.update(observed_path, threshold_factor = resampling_threshold, method = 'mean', resampling_method = 'attractor{}'.format(resample_id), func = model5.conditional_pdf_o)
    apf.compute_error(hidden_path)
    apf_abs_err += apf.abs_error


enkf_abs_err /= num_exprs
bpf_abs_err /= num_exprs
apf_abs_err /= num_exprs
print('EnKF error norm = {}'.format(np.linalg.norm(enkf_abs_err)))
print('BPF error norm = {}'.format(np.linalg.norm(bpf_abs_err)))
print('APF0 error norm = {}'.format(np.linalg.norm(apf_abs_err)))


image_dir = str(script_dir.parent.parent.parent) + '/images/EnKF_vs_BPF/'
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)
t = list(range(s))
ax.plot(t, enkf_abs_err, label = 'avg EnKF abs error')
ax.plot(t, bpf_abs_err, label = 'avg BPF abs error')
ax.plot(t, apf_abs_err, label = 'avg APF0 abs error')
plt.legend()
plt.savefig(image_dir + 'avg_error_{}_{}_{}.png'.format(particle_count, db_id, resample_id))
