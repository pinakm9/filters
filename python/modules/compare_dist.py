import numpy as np
from sklearn.neighbors import NearestNeighbors, KernelDensity
import utility as ut
import tables
import pandas as pd
import matplotlib.pyplot as plt

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
        return sum([np.log(s_k[i]/r_k[i]) for i in range(n)]) * self.dim / n +  np.log(m/(n-1))

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
            ensemble_1 = kde_1.sample(num_samples)
            ensemble_2 = kde_2.sample(num_samples + 7)
            dist_comp = DistComparison(ensemble_1, ensemble_2)
            kl_dist[itr] = dist_comp.compute_KL(k)
        pd.DataFrame(kl_dist).to_csv(saveas if saveas is not None else 'filter_comparison.csv', header=None, index=None)
        plt.figure(figsize = (8, 8))
        x = list(range(iterations))
        plt.scatter(x, kl_dist)
        trend = np.polyfit(x, kl_dist, 1)
        trend = np.poly1d(trend)
        plt.plot(x, trend(x), '--r')
        plt.plot(x, np.zeros(len(x)), color='red')
        if saveas is None:
            saveas = 'filter_comparison.png'
        else:
            saveas = saveas[:-3] + 'png'
        plt.savefig(saveas)
