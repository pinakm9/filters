import simulate as sm
import filter as fl
import numpy as np
import scipy
import plot
"""
A 2D non-linear problem (Henon map)
"""
# Create a Markov chain
a, b, eps  = 0.5, 0.1, 1
mu, id =  np.zeros(2), np.identity(2)
prior = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, id))
process_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, eps*id))
func_h = lambda k, x, noise: np.array([1.0 - a*x[0]**2 + x[1], b*x[0]]) + noise
conditional_pdf_h = lambda k, x, past: scipy.stats.multivariate_normal.pdf(x, mean = func_h(0, past, mu), cov = eps*id)

# Define the observation model
delta = 0.01
H = np.array([[1.0, 1.0],[0.0, 2.0]])
func_o = lambda k, x, noise: np.dot(H, x) + 0.5*np.sin(x) + noise
observation_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, delta*id))
conditional_pdf_o = lambda k, y, condition: scipy.stats.multivariate_normal.pdf(y, mean = func_o(0, condition, mu), cov = delta*id)

# creates a ModelPF object to feed the filter / combine the models
def model(size):
    mc = sm.DynamicModel(size = size, prior = prior, func = func_h, sigma = eps*id, noise_sim = process_noise, conditional_pdf = conditional_pdf_h)
    om = sm.MeasurementModel(size = size, func = func_o, sigma = delta*id, noise_sim = observation_noise, conditional_pdf = conditional_pdf_o)
    return fl.ModelPF(dynamic_model = mc, measurement_model = om), a, b
