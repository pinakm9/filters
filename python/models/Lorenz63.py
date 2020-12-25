# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

# import remaining modules
import simulate as sm
import filter as fl
import numpy as np
import scipy
import plot as plot

"""
A 2D non-linear problem (Henon map)
"""
# set parameters
sigma, rho, beta = 10.0, 28.0, 2.6666666666666665#8.0/3,
eps, delta = 0.0, 0.1
mu, id, zero =  np.zeros(3), np.identity(3), np.zeros(3)
x0 = [-13.793058609513304, -8.198083431274192, 39.06604861256139]

# assign an attractor point as the starting point
def lorenz63_f(t, state):
    x, y, z = state
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

def lorenz_63(x):
    return scipy.integrate.solve_ivp(lorenz63_f, [0.0, 2], x, method='RK45', t_eval=[2]).y.T[0]

"""
print(lorenz_63(x0))
"""

# create a deterministic Markov chain
prior = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(x0, 0.01*id))
process_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, eps*id))
func_h = lambda k, x, noise: lorenz_63(x) + noise
conditional_pdf_h = lambda k, x, past: scipy.stats.multivariate_normal.pdf(x, mean = func_h(k, past, zero), cov = eps*id)

# define the observation model
func_o = lambda k, x, noise: x + noise
observation_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, delta*id))
conditional_pdf_o = lambda k, y, condition: scipy.stats.multivariate_normal.pdf(y, mean = func_o(0, condition, mu), cov = delta*id)

# generates a trajectory according to the dynamic model
def gen_path(length):
    path = np.zeros((length, 3), dtype = 'float64')
    x = x0
    for i in range(length):
        path[i] = x
        x = func_h(i, x, zero)
    return path

# creates a Model object to feed the filter / combine the models
def model(size):
    mc = sm.DynamicModel(size = size, prior = prior, func = func_h, sigma = eps*id, noise_sim = process_noise, conditional_pdf = conditional_pdf_h)
    om = sm.MeasurementModel(size = size, func = func_o, sigma = delta*id, noise_sim = observation_noise, conditional_pdf = conditional_pdf_o)
    return fl.Model(dynamic_model = mc, measurement_model = om)
