import numpy as np
import filter as fl
import scipy
import utility as ut
import matplotlib.pyplot as plt
import plot
import model1


"""
Solution using a gsupf
"""
model = model1.model()
hidden_path = model.hidden_state.generate_path()
observed_path = model.observation.generate_path(hidden_path)

means, covs = model1.update(observed_path[1:])
mu = means[-1]
pf = fl.GlobalSamplingUPF(model, particle_count = 2000, alpha = 1, kappa = 2, beta = 2)
pf.update(observed_path, threshold_factor = 0.1, method = 'mean')
mean = pf.computed_trajectory[-1]

plt.scatter(*zip(*pf.particles))
plt.scatter([mu[0]], [mu[1]], color = 'red')
plt.scatter([mean[0]], [mean[1]], color = 'green')

plt.show()

pf.plot_trajectories(hidden_path, coords_to_plot = [0, 1], show = True)
pf.compute_error(hidden_path)
pf.plot_error(show = True)
print("\n\n error in mean {}\n\n".format(np.linalg.norm(mean - mu)))
