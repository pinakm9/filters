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
s = 2
mu = np.zeros(2)
id = np.identity(d)
# Create a Markov chain
prior = sm.Simulation(target_rv = sm.RVContinuous(name = 'normal', mean = mu, cov = id), \
                      algorithm = lambda *args: np.random.multivariate_normal(mu, id))

f = lambda x: x
mc = sm.GaussianErrorModel(size = s, prior = prior, f = f, sigma = id)

# Define the observation model
f = lambda x: x
om = sm.GaussianObservationModel(size = s, f = f, sigma = id)

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
    for y in Y:
        m_, P_ =  one_step_predict_update(m, P, y)
        m, P = m_, P_
    return m, P

def filering_dist(m, P, Y):
    m_, P_ = n_step_predict_update(m, P, Y)
    return lambda x: scipy.stats.multivariate_normal.pdf(x, mean = m_, cov  = P_), lambda x: scipy.stats.multivariate_normal.cdf(x, mean = m_, cov  = P_), m_, P_

actual_density, actual_cdf, mean, cov = filering_dist(np.array([0.,0.]), id, observed_path)
"""
Solution using a gsupf
"""
pf = fl.GlobalSamplingUPF(model, particle_count = 100)

pf.update(observed_path[1:], threshold_factor = 0.0, method = 'mean')
"""
samples = pf.particles
print("\n\n########## Total Variation ###########\n{}\n#########################################\n\n".format(ut.TV_dist_MC(actual_density, pf.filtering_pdf, pf.particles)))
"""
