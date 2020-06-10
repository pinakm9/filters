import simulate as sm
import numpy as np
import filter as fl
from scipy.stats import multivariate_normal
import utility as ut
"""
 Goal: To compute the distance between the actual filtering distribution and the one given by particle filter
 for a problem with known solution
"""

"""
A 2D problem with known solution
"""
s = 150
# Create a Markov chain
prior = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(np.array([0.,0.]), np.diag([1.,1.])))
f = lambda x: x
G = np.identity(2)
mc = sm.GaussianErrorModel(size = s, prior = prior, f = f, G = G, mu = np.array([0.,0.]), sigma = np.diag([1., 1.]))

# Define the observation model
f = lambda x: x
G = np.identity(2)
om = sm.GaussianObservationModel(size = s, f = f, G = G, mu = np.array([0.,0.]), sigma = 1.0*np.diag([1., 1.]))

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
    m_ += np.dot(K,v)
    P_ -= np.linalg.multi_dot([K, S, K.T])

    return m_, P_

def n_step_predict_update(m, P, Y):
    for y in Y:
        m_, P_ =  one_step_predict_update(m, P, y)
        m, P = m_, P_
    return m, P

def filering_dist(m, P, Y):
    m_, P_ = n_step_predict_update(m, P, Y)
    return lambda x: multivariate_normal.pdf(x, mean = m_, cov  = P_), lambda x: multivariate_normal.cdf(x, mean = m_, cov  = P_), m_, P_

actual_density, actual_cdf, mean, cov = filering_dist(mc.mu, mc.sigma, observed_path)


"""
Solution using a particle filter
"""
pf = fl.ParticleFilter(model, particle_count = 1000)
pf.update(observed_path, threshold_factor = 0.1, method = 'mean')

print(n_step_predict_update(mc.mu, mc.sigma, observed_path))


samples = pf.particles
print(ut.TV_dist_MC(actual_density, pf.filtering_pdf, pf.particles))


"""
Kolmogorov-Smirnov distance
"""
"""
samples = pf.particles[:-1] + np.random.multivariate_normal(mean, cov, 100)
dist = max([abs(actual_cdf(x) - pf.ecdf(x)) for x in samples])
print("Kolmogorov-Smirnov distance : {}".format(dist))
"""

m = np.average(pf.particles, weights = pf.weights, axis = 0)
print(m, mean)
print("error:", pf.compute_error())
