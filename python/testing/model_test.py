import simulate as sm
import numpy as np
import plot

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
