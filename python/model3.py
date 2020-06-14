import simulate as sm
import filter as fl
import numpy as np
import scipy
import plot
"""
A 2D non-linear problem
"""
# Create a Markov chain
s, a, b, eps  = 10, 0.5, 0.1, 1
mu, id =  np.zeros(2), np.identity(2)
prior = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, id))
noise_sim = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, eps*id))
func = lambda k, x, noise: np.array([1.0 - a*x[0]**2 + x[1], b*x[0]]) + noise
conditional_pdf = lambda k, x, past: scipy.stats.multivariate_normal.pdf(x, mean = func(0, past, mu), cov = eps*id)
mc = sm.DynamicModel(size = s, prior = prior, func = func, sigma = eps*id, noise_sim = noise_sim, conditional_pdf = conditional_pdf)

# Define the observation model
delta = 0.01
H = np.array([[1.0, 1.0],[0.0, 2.0]])
f = lambda x: np.dot(x, x)
func = lambda k, x, noise: np.dot(H, x) + 0.5*np.sin(x) + noise
noise_sim = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, delta*id))
conditional_pdf = lambda k, y, condition: scipy.stats.multivariate_normal.pdf(y, mean = func(0, condition, mu), cov = delta*id)
#om = sm.GaussianObservationModel(size = s, f = f, sigma = 0.01*id)
om = sm.Measurement_model(size = s, func = func, sigma = delta*id, noise_sim = noise_sim, conditional_pdf = conditional_pdf)


def model():
    return fl.ModelPF(dynamic_model = mc, measurement_model = om), a, b


"""
hidden_path = mc.generate_path()
observed_path = om.generate_path(hidden_path)
plot.SignalPlotter(signals = [hidden_path, observed_path]).plot_signals(labels = ['hidden', 'observed'], coords_to_plot = [0, 1], show = True)
"""
