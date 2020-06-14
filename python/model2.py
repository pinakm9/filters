import simulate as sm
import filter as fl
import numpy as np
import scipy

"""
A 1D non-linear problem
"""
# Create a Markov chain
x_0 = np.array([1.0])
s, alpha, theta = 10, 3, 0.5
prior = sm.Simulation(algorithm = lambda *args: x_0)
noise_sim = sm.Simulation(algorithm = lambda *args: np.array([np.random.normal(0, 0.01)]))
func = lambda k, x, noise: np.array([1 + np.sin(0.4*np.pi*k)]) + 0.5*x + noise
conditional_pdf = lambda k, x, condition: scipy.stats.norm.pdf(x[0], (np.array([1 + np.sin(0.4*np.pi*k)]) + 0.5*condition)[0], 0.01)
mc = sm.DynamicModel(size = s, prior = prior, func = func, sigma = [0.01], noise_sim = noise_sim, conditional_pdf = conditional_pdf)

# Define the observation model
noise_sim = sm.Simulation(algorithm = lambda* args: np.array([np.random.gamma(a = alpha, scale = theta)]))

def func(k, x, noise):
    if k < 31:
        return np.array([0.2*x[0]**2 + noise])
    else:
        return np.array([0.5*x[0] + noise - 2.0])

def conditional_pdf(k, y, condition):
    if k < 31:
        return scipy.stats.gamma.pdf(y[0], a = alpha, loc = 0.2*condition[0]**2, scale = theta)
    else:
        return scipy.stats.gamma.pdf(y[0], a = alpha, loc = 0.5*condition[0] - 2.0, scale = theta)

om = sm.Measurement_model(size = s, func = func, sigma = [alpha*theta**2], noise_sim = noise_sim, conditional_pdf = conditional_pdf)

hidden_path = mc.generate_path()
observed_path = om.generate_path(hidden_path)
print(observed_path)

def model():
    return 0
