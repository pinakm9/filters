import simulate as sm
import numpy as np
import plot
"""
# Implementation of a Markov chain using only the Simulation class

prior_sim = sm.Simulation(target_rv = None, algorithm = lambda *args: np.random.normal(0.0, 0.5))
sims = [prior_sim]
for i in range(10):
    algorithm = lambda *args: 4*sims[i].current_value + np.random.normal(0.0, 0.01)
    sims.append(sm.Simulation(target_rv = None, algorithm = algorithm))
"""

"""
# Implementation of a Markov chain using StochasticProcess

prior_sim = sm.Simulation(target_rv = None, algorithm = lambda *args: np.random.normal(0.0, 0.5))
sims = [prior_sim]
sims_ = [prior_sim]

alge = lambda:  sims[0].current_value + np.random.normal(0.0, 0.5)
sims.append(sm.Simulation(target_rv = None, algorithm = alge))
alge = lambda:  sims[1].current_value + np.random.normal(0.0, 0.5)
sims.append(sm.Simulation(target_rv = None, algorithm = alge))


for i in range(2):
    alge = lambda : sims_[i].current_value + np.random.normal(0.0, 0.5)
    sims_.append(sm.Simulation(target_rv = None, algorithm = alge))

sp = sm.StochasticProcess(sims)
sp_ = sm.StochasticProcess(sims_)

print(sp.generate_path())
print(sp_.generate_path())
"""
"""
# Create a 2-dimensional Markov chain of form x_k = Ax_(k-1) + q, A = 3x3 matrix, q ~ N(0, sigma) using MarkovChain

# define the prior simulation
prior = sm.Simulation(algorithm = lambda *args: np.random.normal(0.0, 0.5, (2,)))
#prior.algorithm()
# define algorithm for MarkovChain
A = np.array([[1.0, 2.0], [-1.0, 3.0]])

def algorithm(past):
    return np.dot(A, past.current_value) + np.random.normal(0.0, 0.3, (2,))

mc = sm.MarkovChain(size = 10, prior = prior, algorithm = algorithm)
print(mc.generate_path())
print('dim = {}'.format(mc.dimension))

"""
"""
Create a GaussianErrorModel
"""
d = 10
prior = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal([0]*d, np.diag([1]*d)))
f = lambda x: x
G = np.identity(d)
mc = sm.GaussianErrorModel(size = 150, prior = prior, f = f, G = G, mu = [0]*d, sigma = np.diag([1]*d))
paths = mc.generate_paths(5)
pltr = plot.SignalPlotter(signals = paths)
print(pltr.dimension)
pltr.plot_signals(labels=[], coords_to_plot = [1], show = True)
