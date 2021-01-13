import numpy as np
from sklearn.neighbors import NearestNeighbors, KernelDensity
import utility as ut
import tables
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


class DistComparison:
    """
    Class for comparing distributions
    """
    def __init__(self, ensemble_1, ensemble_2):
        self.ensemble_1 = ensemble_1
        self.ensemble_2 = ensemble_2
        self.dim = len(ensemble_1[0])

    @ut.timer
    def compute_KL(self, k):
        n = len(self.ensemble_1)
        m = len(self.ensemble_2)
        # compute p_hat
        neigh = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(self.ensemble_1)
        dist, _ = neigh.kneighbors(self.ensemble_1)
        r_k = [dist[i][k] for i in range(n)]
        #print(dist.shape)
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # train for getting nearest neighbour
        neigh.fit(self.ensemble_2)
        dist, _ = neigh.kneighbors(self.ensemble_1)
        s_k = [dist[i][k - 1] for i in range(n)]
        #print(dist.shape)
        return sum([np.log(s_k[i]/r_k[i]) for i in range(n)]) * self.dim / n +  self.dim * np.log(m/(n-1))

class PFComparison:
    def __init__(self, assml_file_1, assnl_file_2):
        self.file_1 = assml_file_1
        self.file_2 = assnl_file_2

    @ut.timer
    def compare(self, num_samples, k, saveas=None):
        hdf5_1 = tables.open_file(self.file_1, 'r')
        hdf5_2 = tables.open_file(self.file_2, 'r')
        iterations = len(hdf5_1.root.observation.read().tolist())
        kl_dist = np.zeros(iterations, dtype=np.float32)
        for itr in range(iterations):
            ensemble_1 = np.array(getattr(hdf5_1.root.particles, 'time_' + str(itr)).read().tolist())
            ensemble_2 = np.array(getattr(hdf5_2.root.particles, 'time_' + str(itr)).read().tolist())
            weights_1 = np.array(getattr(hdf5_1.root.weights, 'time_' + str(itr)).read().tolist()).flatten()
            weights_2 = np.array(getattr(hdf5_2.root.weights, 'time_' + str(itr)).read().tolist()).flatten()
            # remove zero weights
            idx_1 = np.where(weights_1 > 1e-10)
            idx_2 = np.where(weights_2 > 1e-10)
            kde_1 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(ensemble_1[idx_1], sample_weight=weights_1[idx_1])
            kde_2 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(ensemble_2[idx_2], sample_weight=weights_2[idx_2])
            ensemble_1 = kde_1.sample(num_samples + 1)
            ensemble_2 = kde_2.sample(num_samples)
            dist_comp = DistComparison(ensemble_1, ensemble_2)
            kl_dist[itr] = dist_comp.compute_KL(k)
        hdf5_1.close()
        hdf5_2.close()
        pd.DataFrame(kl_dist).to_csv(saveas + '.csv' if saveas is not None else 'filter_comparison.csv', header=None, index=None)
        plt.figure(figsize = (8, 8))
        x = np.array(list(range(iterations)))
        idx_1 = np.where(kl_dist >= 0.0)
        idx_2 = np.where(kl_dist < 0.0)
        x_, y_ = x[idx_1], kl_dist[idx_1]
        plt.scatter(x_, y_, color='blue')
        plt.scatter(x[idx_2], kl_dist[idx_2], color='red')
        """
        try:
            ransac = linear_model.RANSACRegressor()
            x__ = x_.reshape((-1, 1))
            ransac.fit(x__, y_)
            plt.plot(x_, ransac.predict(x__), '--r', color='blue', label='RANSAC trend (computed from blue dots)')
        except:
            x_2 = (x_**2).sum()
            x_1 = x_.sum()
            A = np.array([[x_2, x_1], [x_1, len(x_)]])
            a, b = np.linalg.solve(A, [(x_*y_).sum(), y_.sum()])
            plt.plot(x_, a*x_ + b, '--r', color='blue', label='linear trend (computed from blue dots)')
        #"""
        plt.plot(x, np.zeros(len(x)), color='green', label='x-axis')
        plt.xlabel('assimilation step')
        plt.ylabel('approximate KL divergence')
        plt.title('{} vs {}'.format(self.file_1[:-3], self.file_2[:-3]))
        plt.legend()
        if saveas is None:
            saveas = 'filter_comparison.png'
        else:
            saveas = saveas + '.png'
        plt.savefig(saveas)
