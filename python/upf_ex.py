import numpy as np
import filter as fl
import scipy
import utility as ut
import matplotlib.pyplot as plt
import plot

@ut.timer
def experiment(model, particle_counts, num_exprs, alpha, beta, kappa, resampling_threshold, titles = None, file_paths = None, exact_mean = None):
    # compute average error for each particle count
    avg_abs_err, err_in_mean = [], []
    for particle_count in particle_counts:
        error, err_m = 0.0, 0.0
        for expr in range(num_exprs):
            print('\rusing {:04} particle(s): experiment# {:03d}'.format(particle_count, expr), end = '')
            # Generate paths
            hidden_path = model.hidden_state.generate_path()
            observed_path = model.observation.generate_path(hidden_path)
            # create particle filter
            pf = fl.GlobalSamplingUPF(model, particle_count = particle_count, alpha = alpha, beta = beta, kappa = kappa)
            pf.update(observed_path , threshold_factor = resampling_threshold, method = 'mean')
            pf.compute_error(hidden_path)
            error += np.linalg.norm(pf.error[-1])
            if exact_mean is not None:
                err_m += np.linalg.norm(exact_mean(observed_path[1:]) - pf.computed_trajectory[-1])
        avg_abs_err.append(error/num_exprs)
        if exact_mean is not None:
            err_in_mean.append(err_m/num_exprs)

    print(avg_abs_err)
    # plot average error vs particle_count
    x = np.linspace(min(particle_counts), max(particle_counts), 100)
    y = scipy.interpolate.make_interp_spline(particle_counts, avg_abs_err)(x)
    plt.plot(x, y)
    plt.xlabel('number of particles')
    plt.ylabel('average absolute error')
    if titles is not None:
        plt.title(titles[0])
    if file_paths is not None:
        plt.savefig(file_paths[0])

    # plot error in mean vs particle_count
    if exact_mean is not None:
        y = scipy.interpolate.make_interp_spline(particle_counts, err_in_mean)(x)
        plt.plot(particle_counts, err_in_mean)
        plt.xlabel('number of particles')
        plt.ylabel('error in mean')
        if titles is not None and len(titles) > 0:
            plt.title(titles[1])
        if file_paths is not None and len(file_paths) > 0:
            plt.savefig(file_paths[1])
    return avg_abs_err, err_in_mean
