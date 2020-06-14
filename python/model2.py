import simulate as sm
import filter as fl
import numpy as np
import scipy
import plot
"""
A 1D non-linear problem
"""
# Create a Markov chain
x_0 = np.array([1.0])
s, alpha, theta, w  = 10, 3, 0.5, 0.04
prior = sm.Simulation(algorithm = lambda *args: x_0)
noise_sim = sm.Simulation(algorithm = lambda* args: np.array([np.random.gamma(shape = alpha, scale = theta)]))
func = lambda k, x, noise: np.array([1.0 + np.sin(w*np.pi*k)]) + 0.5*x + noise
conditional_pdf = lambda k, x, past: scipy.stats.gamma.pdf(x[0], a = alpha, loc = (np.array([1.0 + np.sin(w*np.pi*k)]) + 0.5*past)[0], scale = theta)
mc = sm.DynamicModel(size = s, prior = prior, func = func, sigma = np.array([[alpha*theta**2]]), noise_sim = noise_sim, conditional_pdf = conditional_pdf)

# Define the observation model
sigma, two, zero, threshold = 0.00, np.array([2.0]), np.array([0.0]), 10
noise_sim = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal([0.0], [[sigma]]))
f1 = lambda x, noise: x + 0.0*np.cos(x**2) + noise
f2 = lambda x, noise: 2*x + noise
def func(k, x, noise):
    if k < threshold:
        return f1(x, noise)
    else:
        return f2(x, noise)

def conditional_pdf(k, y, condition):
    if k < threshold:
        return scipy.stats.multivariate_normal.pdf(y, f1(condition, zero), [[sigma]])
    else:
        return scipy.stats.multivariate_normal.pdf(y, f2(condition, zero), [[sigma]])

om = sm.Measurement_model(size = s, func = func, sigma = np.array([[sigma]]), noise_sim = noise_sim, conditional_pdf = conditional_pdf)

def model():
    return fl.ModelPF(dynamic_model = mc, measurement_model = om)


#"""
hidden_path = mc.generate_path()
observed_path = om.generate_path(hidden_path)
plot.SignalPlotter(signals = [hidden_path, observed_path]).plot_signals(labels = ['hidden', 'observed'], coords_to_plot = [0], show = True)
#"""
