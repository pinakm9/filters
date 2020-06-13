import simulate as sm
import numpy as np
import filter as fl
import scipy
import utility as ut
import matplotlib.pyplot as plt
import plot
"""
Goal: To compute the distance between the actual filtering distribution and the one given by particle filter
for a problem with known solution
"""

"""
A 2D problem with known solution
"""
d = 2
s = 50
mu = np.zeros(2)
id = np.identity(d)
# Create a Markov chain
prior = sm.Simulation(target_rv = sm.RVContinuous(name = 'normal', mean = mu, cov = id), \
                      algorithm = lambda *args: np.random.multivariate_normal(mu, id))

f = lambda x: x
mc = sm.GaussianErrorModel(size = s, prior = prior, f = f, sigma = id)

# Define the observation model
f = lambda x: x
om = sm.GaussianObservationModel(size = s, f = f, sigma = 0.01*id)

# create a ModelPF object to feed the filter / combine the models
model = fl.ModelPF(dynamic_model = mc, measurement_model = om)

# Generate paths
hidden_path = model.hidden_state.generate_path()
observed_path = model.observation.generate_path(hidden_path)
"""
Solution to the filtering problem
"""
def one_step_predict_update(m, P, y):
    # predict mean and covariance
    m_ = m
    P_ = P + mc.sigma

    # update mean and covariance
    v = y - m_
    S = P_ + om.sigma
    K = np.dot(P_, np.linalg.inv(S))
    #print("\n~~~~~~~~~~~~~ K = \n{}\n{}\n{}\n{}\n ~~~~~~~~~~~~~~~~\n".format(K, P, S, om.sigma))
    m_ += np.dot(K,v)
    P_ -= np.linalg.multi_dot([K, S, K.T])

    return m_, P_

def n_step_predict_update(m, P, Y):
    m_, P_ = m, P
    for y in Y:
        m_, P_ =  one_step_predict_update(m_, P_, y)
    return m_, P_

def filering_dist(m, P, Y):
    m_, P_ = n_step_predict_update(m, P, Y)
    return lambda x: scipy.stats.multivariate_normal.pdf(x, mean = m_, cov  = P_), lambda x: scipy.stats.multivariate_normal.cdf(x, mean = m_, cov  = P_), m_, P_

actual_density, actual_cdf, mean, cov = filering_dist(mu, id, observed_path[1:])



"""
Solution using a gsupf
"""
pf = fl.GlobalSamplingUPF(model, particle_count = 100, alpha = 1, kappa = 2, beta = 0)

pf.update(observed_path , threshold_factor = 0.1, method = 'mean')


samples = pf.particles
print("\n\n########## Total Variation ###########\n{}\n#############\n\n".format(ut.TV_dist_MC(actual_density, pf.filtering_pdf, pf.particles)))


m = np.average(pf.particles, weights = pf.weights, axis = 0)
print("\n\n Mean distance {}\n\n".format(np.linalg.norm(m-mean)))
# plot.SignalPlotter(signals = [hidden_path, observed_path]).plot_signals(coords_to_plot = [0], show = True)
plt.scatter(*zip(*pf.particles))
plt.scatter([m[0]], [m[1]], color = 'red')
plt.scatter([mean[0]], [mean[1]], color = 'green')
plt.show()
pf.plot_trajectories(hidden_path, coords_to_plot = [0, 1], show = True)
pf.compute_error(hidden_path)
pf.plot_error(show = True)
