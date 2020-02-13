# particle filter test
import filter as fl
import numpy as np
import simulate as sm
import plot
import matplotlib.pyplot as plt
import utility as ut
#np.random.seed(seed = 1)
rho = 2

@ut.timer
def collapse(n, d): # n -> number of particles, d -> dimension of the problem
    # set parameters
    mu = [1.0]*d
    sigma = np.diag([1.0]*d)

    # create a dynamic model
    prior = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, sigma))
    f = lambda x: x
    G = np.identity(d)
    dynamic_model = sm.GaussianErrorModel(size = 2, prior = prior, f = f, G = G, mu = mu, sigma = sigma)

    # create a measurement_model

    f = lambda x: x
    G = rho*np.identity(d)
    measurement_model = sm.GaussianObservationModel(conditions = dynamic_model.sims, f = f, G = G, mu = mu, sigma = sigma)

    # create a ModelPF object to feed the filter
    model = fl.ModelPF(dynamic_model = dynamic_model, measurement_model = measurement_model)
    """
    print(model.hidden_state.generate_paths(2))
    print(model.observation.generate_paths(2))
    """
    # figure out gradient to pass to QuadraticImplicitPF
    def grad(x, y, x_0):
        return (x - x_0) + (x - y)

    L = np.sqrt(1.0/(1 + rho**2))*np.identity(d)
    H = (1 + rho**2)*np.identity(d)

    def hessian(x, y, x_0):
        return H

    def cholesky_factor_invT(x, y, x_0):
        return L
    # construct a ParticleFilter object
    pf = fl.RandomQuadraticIPF(model, n)
    hidden = model.hidden_state.generate_path()
    signal = model.observation.generate_path()
    pf.update(signal, threshold_factor = 0.0, method = 'mean')

    #"""
    #plot.SignalPlotter(signals = [signal, pf.hidden_trajectory, hidden]).plot_signals( labels = ['observation', 'hidden', 'original'],\
    #                    coords_to_plot = [9], show = True)
    return np.max(pf.weights)

itr = 100
for n in [300]:
    for d in [10, 50, 100]:
        max_w = []
        for i in range(itr):
            print('iteration = {}:'.format(i))
            max_w.append(collapse(n,d))
        plt.title("(d, n) = ({}, {})".format(d, n))
        plt.xlabel("maximum weight")
        plt.hist(max_w, 15, color = "darkblue")
        plt.savefig("../images/rho6/ipf_max_weight_{}_{}_{}_{}.png".format(d, n, itr, rho))
#plt.show()
