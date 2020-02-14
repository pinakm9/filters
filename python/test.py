# particle filter test
import filter as fl
import numpy as np
import simulate as sm
import plot
import matplotlib.pyplot as plt
import utility as ut
#np.random.seed(seed = 1)
rho = 1

@ut.timer
def collapse(n, d): # n -> number of particles, d -> dimension of the problem
    # set parameters
    mu = [0.0]*d
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
    # construct a ParticleFilter object
    hidden = model.hidden_state.generate_path()
    signal = model.observation.generate_path()
    #plot.SignalPlotter(signals = [signal, hidden]).plot_signals( labels = ['observation', 'original'], coords_to_plot = [1,9], show = True)
    pf = fl.ParticleFilter(model, n)
    weights = pf.update(signal, threshold_factor = 0.0, method = 'mean')
    """
    print('hidden', pf.computed_trajectory)
    print('observation', signal)
    """
    err = pf.compute_error()
    print("Error mean mean= {}, Error mean standard deviation = {}".format(np.mean(err[1]), np.mean(err[2])))
    #plot.SignalPlotter(signals = [signal, pf.computed_trajectory, hidden]).plot_signals( labels = ['observation', 'hidden', 'original'], coords_to_plot = [1,9], show = True)
    return np.max(pf.weights)

itr = 100
for n in [500]:
    for d in [10, 50, 100]:
        max_w = []
        for i in range(itr):
            print('iteration = {}:'.format(i))
            max_w.append(collapse(n,d))
        plt.title("(d, n) = ({}, {})".format(d, n))
        plt.xlabel("maximum weight")
        plt.hist(max_w, bins = 15, color = 'blue')
        plt.savefig("../images/max_weight_{}_{}_{}_{}.png".format(d, n, itr, rho))
#plt.show()
"""
# step - 1 : construct a ModelPF object as the filter's input
prior = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal([0]*10, np.diag([1,2,3,2,1,1,1,1,1,1])))
f = lambda x: 2*x
G = np.random.normal(size = (10,10))
mc = sm.GaussianErrorModel(size = 20, prior = prior, f = f, G = G, mu = [1]*10, sigma = np.diag([1,7,8,7,1,1,1,1,1,1]))
# construct the dynamic algorithm
def dynamic_algorithm(past):
    return past.current_value + np.random.normal(0.0, 0.7)
# construct the measurement_algorithm:
def measurement_algorithm(condition):
    return condition.current_value + np.random.normal(0.0, 0.2)
# construct the dynamic pdf
def dynamic_pdf(x, y):
    return norm.pdf(x-y, scale = 0.5)
# construct the measurement pdf
def measurement_pdf(y, x):
    return norm.pdf(y-x, scale = 0.2)
# construct model
model = fl.ModelPF(30,  prior, dynamic_algorithm, measurement_algorithm, dynamic_pdf, measurement_pdf)
#print(model.hidden_state.generate_paths(2))
#print(model.observation.generate_paths(2))

# step - 2: construct a particle filter object
pf = fl.ParticleFilter(model, 50)
hidden = model.hidden_state.generate_path()
signal = model.observation.generate_path()
weights = pf.update(signal)
pf.compute_hidden_state(method = 'mode')
print('hidden', pf.hidden_state)
print('observation', signal)
pf.plot_signals(signals = [signal, pf.hidden_state, hidden], labels = ['observation', 'hidden', 'original'], coords_to_plot = [5])
"""
