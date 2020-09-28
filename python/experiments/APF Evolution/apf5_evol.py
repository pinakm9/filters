"""
Plots evolution of ensembles
"""
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
import filter as fl
import model5
import config as cf
import attract as atr

# set parameters
ev_time = int(sys.argv[1])
model, a, b, x0 = model5.model(size = ev_time)
db_id = '14_3_medium'
resample_id = '0'
particle_count = 100
resampling_threshold = 0.1
db_path = str(script_dir.parent.parent.parent) + '/data/henon_attractor_{}.h5'.format(db_id)
attractor_sampler = atr.AttractorSampler(db_path = db_path)

# set up configuration
config = {'Henon a': a, 'Henon b': b, 'Particle count': particle_count, 'Resampling threshold': resampling_threshold,\
          'Evolution time': ev_time, 'Model': 'model5', 'Starting point': x0, 'Database Id': db_id, 'Attractor resampling scheme': resample_id}
cc = cf.ConfigCollector(expr_name = 'Evolution', folder = script_dir)
cc.add_params(config)
cc.write()

# assimilation using a bootstrap particle filter
hidden_path = model5.gen_path(length = ev_time)
observed_path = model.observation.generate_path(hidden_path)
apf = fl.AttractorPF(model, particle_count = particle_count, attractor_sampler = attractor_sampler, record_path = cc.res_path + '/apf_assimilation.h5')
apf.update(observed_path, threshold_factor = resampling_threshold, method = 'mean', resampling_method = 'attractor{}'.format(resample_id),\
           func = model5.conditional_pdf_o)
apf.plot_ensembles(hidden_path, obs_inv = True, pt_size = 80)
