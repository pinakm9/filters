import simulate as sm
import filter as fl
import numpy as np
import scipy
import plot
"""
A 1D non-linear problem
"""
# creates a ModelPF object to feed the filter / combine the models
def model(size):
    # Create a Markov chain
    x_0 = np.array([1.0])
    alpha, theta, w  = 3, 0.5, 0.04
    prior = sm.Simulation(algorithm = lambda *args: x_0)
    process_noise = sm.Simulation(algorithm = lambda* args: np.array([np.random.gamma(shape = alpha, scale = theta)]))
    func_h = lambda k, x, noise: np.array([1.0 + np.sin(w*np.pi*k)]) + 0.5*x + noise
    conditional_pdf_h = lambda k, x, past: scipy.stats.gamma.pdf(x[0], a = alpha, loc = (np.array([1.0 + np.sin(w*np.pi*k)]) + 0.5*past)[0], scale = theta)

    # Define the observation model
    sigma_o, two, zero, threshold = 0.0001, np.array([2.0]), np.array([0.0]), int(size/2)
    observation_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal([0.0], [[sigma_o]]))
    f1 = lambda x, noise: 0.2*x**2 + noise
    f2 = lambda x, noise: 0.5*x + noise - two
    def func_o(k, x, noise):
        #print('k = {}'.format(k))
        if k < threshold:
            return f1(x, noise)
        else:
            return f2(x, noise)
    def conditional_pdf_o(k, y, condition):
        if k < threshold:
            return scipy.stats.multivariate_normal.pdf(y, f1(condition, zero), [[sigma_o]])
        else:
            return scipy.stats.multivariate_normal.pdf(y, f2(condition, zero), [[sigma_o]])

    mc = sm.DynamicModel(size = size, prior = prior, func = func_h, sigma = np.array([[alpha*theta**2]]),\
                        noise_sim = process_noise, conditional_pdf = conditional_pdf_h)
    om = sm.MeasurementModel(size = size, func = func_o, sigma = np.array([[sigma_o]]),\
                        noise_sim = observation_noise, conditional_pdf = conditional_pdf_o)
    return fl.ModelPF(dynamic_model = mc, measurement_model = om)
