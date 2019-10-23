import filter as fl
import numpy as np
from scipy.stats import norm
import simulate as sm
"""
# signal-plotting test
size = 100
signal =  np.array([(t, t**2, t**3, t**4) for t in range(size)])
noise = np.random.normal(0.0, 1, size = (size, len(signal[0])))
observation = signal + noise
flt = fl.Filter(signal)
flt.plot_signals(signals = [signal, observation], labels = ['actual', 'observed'], coords_to_plot = [2, 1])
"""

# particle filter test

# step - 1 : construct a ModelPF object as the filter's input
# construct the prior
prior = sm.Simulation(target_rv = None, algorithm = lambda *args: np.random.normal(0.0, 0.5))
# construct the dynamic algorithm
def dynamic_algorithm(past):
    return past.current_value + np.random.normal(0.0, 0.7)
# construct the measurement_algorithm:
def measurement_algorithm(condition):
    return condition.current_value + np.random.normal(0.0, 0.2)
# construct the dynamic pdf
def dynamic_pdf(x, y):
    return norm.pdf(x-y, scale = 0.5)
# construct the measurement pdf
def measurement_pdf(y, x):
    return norm.pdf(y-x, scale = 0.2)
# construct model
model = fl.ModelPF(30,  prior, dynamic_algorithm, measurement_algorithm, dynamic_pdf, measurement_pdf)
#print(model.hidden_state.generate_paths(2))
#print(model.observation.generate_paths(2))

# step - 2: construct a particle filter object
pf = fl.ParticleFilter(model, 100)
hidden = model.hidden_state.generate_path()
signal = model.observation.generate_path()
weights = pf.update(signal)
pf.compute_hidden_state(method = 'mode')
print('hidden', pf.hidden_state)
print('observation', signal)
pf.plot_signals(signals = [signal, pf.hidden_state, hidden], labels = ['observation', 'hidden', 'original'])
