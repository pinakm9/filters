import simulate as sm
import filter as fl
import numpy as np
import scipy

"""
A 1D non-linear problem
"""
# Create a Markov chain
x_0 = np.array([1.0])
s = 10
prior = sm.Simulation(algorithm = lambda *args: x_0)
noise_sim = sm.Simulation(algorithm = lambda *args: np.array([np.random.normal(0, 0.01)]))
func = lambda k, x, noise: np.array([1 + np.sin(0.4*np.pi*k)]) + 0.5*x + noise
conditional_pdf = lambda k, x, condition: scipy.stats.norm.pdf(x[0], (np.array([1 + np.sin(0.4*np.pi*k)]) + 0.5*condition)[0], 0.01)
mc = sm.DynamicModel(size = s, prior = prior, func = func, sigma = [0.01], noise_sim = noise_sim, conditional_pdf = conditional_pdf)
#print(mc.generate_path())
