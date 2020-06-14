import simulate as sm
import filter as fl
import numpy as np
import scipy

"""
A 1D non-linear problem
"""
# Create a Markov chain
x_0 = np.array([1.0])
s, alpha, theta  = 60, 3, 2
prior = sm.Simulation(algorithm = lambda *args: x_0)
noise_sim = sm.Simulation(algorithm = lambda* args: np.array([np.random.gamma(shape = alpha, scale = theta)]))
func = lambda k, x, noise: np.array([1 + np.sin(0.4*np.pi*k)]) + 0.5*x + noise
conditional_pdf = lambda k, x, past: scipy.stats.gamma.pdf(x[0], a = alpha, loc = (np.array([1 + np.sin(0.4*np.pi*k)]) + 0.5*past)[0], scale = theta)
mc = sm.DynamicModel(size = s, prior = prior, func = func, sigma = np.array([[alpha*theta**2]]), noise_sim = noise_sim, conditional_pdf = conditional_pdf)

# Define the observation model
sigma = 0.1
noise_sim = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal([0.0], [[sigma]]))

def func(k, x, noise):
    if True:#k < 31:
        return 0.2*x + noise
    #else:
        #return 0.5*x + noise - np.array([2.0])

def conditional_pdf(k, y, condition):
    if True:#k < 31:
        return scipy.stats.multivariate_normal.pdf(y, 0.2*condition, [[sigma]])
    #else:
        #return scipy.stats.multivariate_normal.pdf(y, 0.5*condition - np.array([2.0]), [[sigma]])

om = sm.Measurement_model(size = s, func = func, sigma = np.array([[sigma]]), noise_sim = noise_sim, conditional_pdf = conditional_pdf)

"""
hidden_path = mc.generate_path()
observed_path = om.generate_path(hidden_path)
print(observed_path)
"""

def model():
    return fl.ModelPF(dynamic_model = mc, measurement_model = om)
