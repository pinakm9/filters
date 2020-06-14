import simulate as sm
import filter as fl
import numpy as np
import scipy

"""
A 2D problem with known solution
"""
d = 2
s = 10
mu = np.zeros(2)
id = np.identity(d)
# Create a Markov chain
prior = sm.Simulation(target_rv = sm.RVContinuous(name = 'normal', mean = mu, cov = id), \
                      algorithm = lambda *args: np.random.multivariate_normal(mu, id))

A = np.array([[1.0, 1.0],[0.0, -2.0]])
f = lambda x: np.dot(A, x)
mc = sm.GaussianErrorModel(size = s, prior = prior, f = f, sigma = id)

# Define the observation model
H = np.array([[1.0, 1.0],[0.0, 2.0]])
f = lambda x: np.dot(H, x)
om = sm.GaussianObservationModel(size = s, f = f, sigma = 0.01*id)

"""
Solution to the filtering problem
"""
def one_step_predict_update(m, P, y):
    # predict mean and covariance
    m_ = np.dot(A, m)
    P_ = np.linalg.multi_dot([A, P, A.T]) + mc.sigma
    # update mean and covariance
    v = y - np.dot(H, m_)
    S = np.linalg.multi_dot([H, P_, H.T]) + om.sigma
    K = np.linalg.multi_dot([P_, H.T, np.linalg.inv(S)])
    #print("\n~~~~~~~~~~~~~ K = \n{}\n{}\n{}\n{}\n ~~~~~~~~~~~~~~~~\n".format(K, P, S, om.sigma))
    m_ += np.dot(K,v)
    P_ -= np.linalg.multi_dot([K, S, K.T])
    return m_, P_

def update(Y, m0 = mu, P0 = id):
    m, P = m0, P0
    means = [m]
    covs = [P]
    for y in Y:
        m, P =  one_step_predict_update(m, P, y)
        means.append(m)
        covs.append(P)
    return np.array(means), np.array(covs)

# create a ModelPF object to feed the filter / combine the models
def model():
    return fl.ModelPF(dynamic_model = mc, measurement_model = om)
