# Classes defining generic signal-processing filters
import numpy as np
import scipy
import utility as ut
import simulate as sm
import collections as cl
import scipy.optimize as opt
import plot

class ModelPF():
    """
    Description:
        A class for defining dynamic and measurement models for a particle filter.
    Attributes:
        hidden_state: a MarkovChain object simulating the hidden state
        observation: an SPConditional object simulating the observations
    """
    def __init__(self, size = None, prior = None, dynamic_algorithm = None, measurement_algorithm = None, dynamic_pdf = None, measurement_pdf = None, dynamic_model = None, measurement_model = None):
        """
        Args:
            size: length of the MarkovChain self.hidden_state
            prior: a Simulation object for defining the MarkovChain self.hidden_state
            dynamic_algorithm: algorithm for defining self.hidden_state
            measurement_algorithm: algorithm for defining self.observation
            dynamic_pdf: p(x_k|x_(k-1)), x_k is the hidden state
            measurement_pdf: p(y_k|x_k), y_k is the observation
            dynamic_model: MarkovChain object specifying the hidden_state model
            measurement_model: SPConditional object specifying the observation model
        """
        # create the Markov chain of hidden state X_t and observation Y_t if the models are not explicitly specified
        if dynamic_model is None:
            self.hidden_state = sm.MarkovChain(size = size, prior = prior, algorithm = dynamic_algorithm, conditional_pdf = dynamic_pdf)
        else:
            self.hidden_state = dynamic_model
        if measurement_model is None:
            self.observation = sm.SPConditional(conditions = self.hidden_state.sims, algorithm = measurement_algorithm, conditional_pdf = measurement_pdf)
        else:
            self.observation = measurement_model


class ParticleFilter():
    """
    Description:
         A class for defining particle filters
         Parent class: Filter
    Attributes (extra):
        model: a ModelPF object containing the dynamic and measurement models
        particles: particles used to estimate the filtering distribution
        particle_count: number of particles
        weights: weights computed by the particle filter
        current_time: integer-valued time starting at 0 denoting index of current hidden state
    """
    def __init__(self, model, particle_count, save_trajectories = False):
        """
        Args:
            model: a ModelPF object containing the dynamic and measurement models
            particle_count: number of particles to be used
            start_time: time at first obeservation, default = 0.0
            time_step: time step between consecutive observations, default = 1.0
        """
        self.model = model
        # draw self.particle_count samples from the prior distribution and reshape for hstacking later
        self.particles = []
        self.particle_count = particle_count
        self.weights = np.ones(particle_count)/particle_count
        self.current_time = 0

        self.save_trajectories = save_trajectories
        self.true_trajectory = self.model.hidden_state.current_path
        # figure out the dimension of the problem
        sample = self.model.hidden_state.sims[0].algorithm()
        if np.isscalar(sample):
            self.dimension = 1
        else:
            self.dimension = len(sample)
        self.computed_trajectory = np.empty((0, self.dimension))
        if save_trajectories:
            self.trajectories = [self.particles]
        #super().__init__(dimension = self.model.hidden_state.sims[0].dimension, start_time = start_time, time_step = time_step)

    def compute_weights(self, observation):
        """
        Description:
            Updates weights according to the last observation
        Args:
            observation: an observation of dimension = self.dimension
        Returns:
            self.weights
        """
        # predict the new particles
        if self.current_time > 0:
            self.particles = [self.model.hidden_state.sims[self.current_time].generate(1, particle)[0] for particle in self.particles]
        else:
            self.particles = self.model.hidden_state.sims[0].generate(self.particle_count)

        # compute new weights
        for i in range(self.particle_count):
            #prob1 = self.model.hidden_state.conditional_pdf(self.current_time, new_particles[i], self.particles[i])
            prob2 = self.model.observation.conditional_pdf(self.current_time, observation, self.particles[i])
            #prob3 = self.importance_pdf(new_particles[i], self.particles[i])
            self.weights[i] *= prob2
        # print(self.weights.sum(), np.max(self.weights))
        # normalize weights
            self.weights /= self.weights.sum()

        if self.save_trajectories:
            self.trajectories = np.append(self.trajectories, [self.particles], axis = 0)
        return self.weights


    def systematic_resample(self):
        """ Performs the systemic resampling algorithm used by particle filters.
        This algorithm separates the sample space into N divisions. A single random
        offset is used to to choose where to sample from for all divisions. This
        guarantees that every sample is exactly 1/N apart.
        Parameters
        ----------
        weights : list-like of float
            list of weights as floats
        Returns
        -------
        indexes : ndarray of ints
            array of indexes into the weights defining the resample. i.e. the
            index of the zeroth resample is indexes[0], etc.
        """

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (np.random.random() + np.arange(self.particle_count)) / self.particle_count

        indices = np.zeros(self.particle_count, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.particle_count:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        self.particles = np.array([self.particles[i] for i in indices])
        self.weights = np.ones(self.particle_count)/self.particle_count
        return len(np.unique(indices))

    def resample(self, threshold_factor = 0.1):
        """
        Description:
            Performs resampling
        Args:
            threshold_factor: fraction of self.particle_count which the effective particle count has to surpass in order to stop resampling (0 implies no resampling)
        Returns:
            bool, True if resampling occurred, False otherwise
        """
        # resample if effective particle count criterion is met
        if 1.0/(self.weights**2).sum() < threshold_factor*self.particle_count:
            indices = np.random.choice(self.particle_count, self.particle_count, p = self.weights)
            u = self.systematic_resample()#np.take(a = self.particles, indices = indices, axis = 0)
            #print("\n\n Num particles = {} \n\n".format(u))
            #print("resampled weights:", self.weights.sum(), np.max(self.weights))
            # create weight map for faster computation
            """
            index_map = dict(cl.Counter(indices))
            self.weight_map = np.zeros((len(index_map), 2))
            for i, (key, value) in enumerate(index_map.items()):
                self.weight_map[i] = [key, value*self.weights[0]]
            """
            return True # resampling occurred
        return False # resampling didn't occur

    def filtering_pdf(self, x):
        """
        Description:
            Computes the filtering distribution pi(x_k|y_(1:k))
        Args:
            x: input
            time: time at which to compute the filtering distribution, same as k in the description
        Returns:
            value of the pdf at x
        """
        result = 0.0
        for i in range(self.particle_count):
            result += self.weights[i]*ut.delta(x, self.particles[i])
        return result

    def compute_trajectory(self, method = 'mean'):
        """
        Description:
            Computes hidden trajectory
        """
        if method == 'mode':
            # for each time find the most likely particle
            new_hidden_state = self.particles[np.array(list(map(self.filtering_pdf, self.particles))).argmax()]
            self.computed_trajectory = np.append(self.computed_trajectory, [new_hidden_state], axis = 0)
        elif method == 'mean':
            new_hidden_state = np.average(self.particles, weights = self.weights, axis = 0)
            self.computed_trajectory = np.append(self.computed_trajectory, [new_hidden_state], axis = 0)
        return self.computed_trajectory

    def update(self, observations, threshold_factor = 0.1, method = 'mean'):
        """
        Description:
            Updates using all the obeservations using self.compute_weights and self.resample
        Args:
            observations: list/np.array of observations to pass to self.compute_weights
            threshold_factor: fraction of self.particle_count which the effective particle count has to surpass in order to stop resampling
        Returns:
            self.weights
        """
        self.current_time = 0
        self.observed_path = observations
        for observation in self.observed_path:
            self.compute_weights(observation = observation)
            self.resample(threshold_factor = threshold_factor)
            if method is not None:
                self.compute_trajectory(method = method)
                self.current_time += 1
        return self.weights

    def compute_error(self, hidden_path):
        """
        Description:
            Computes error in assimilation for a random path
        """
        self.error = hidden_path - self.computed_trajectory
        self.rmse = np.linalg.norm(self.error)/np.sqrt(len(hidden_path))
        self.error_mean = np.mean(self.error, axis = 0)
        self.error_cov = np.std(self.error, axis = 0)

    def plot_trajectories(self, hidden_path, coords_to_plot, show = False, file_path = None, title = None):
        plot.SignalPlotter(signals = [hidden_path, self.observed_path, self.computed_trajectory])\
            .plot_signals(labels = ['hidden', 'observed', 'computed'], styles = [{'linestyle':'solid'}, {'marker':'x'}, {'marker':'o'}],\
            plt_fns = ['plot', 'scatter', 'scatter'], colors = ['black', 'blue', 'red'], coords_to_plot = coords_to_plot,\
            show = show, file_path = file_path, title = title)

    def plot_error(self, show = False, file_path = None, title = None):
        plot.SignalPlotter(signals = [abs(self.error)]).plot_signals(labels = ['absolute error'], styles = [{'linestyle':'solid'}],\
            plt_fns = ['plot'], colors = ['black'], coords_to_plot = [0], show = show, file_path = file_path, title = title)



class GlobalSamplingUPF(ParticleFilter):
    """
    Description:
         A class for defining unscented particle filters
         Parent class: ParticleFilter
    Attributes (extra):
    """
    def __init__(self, model, particle_count, alpha = 0.1, beta = 2.0, kappa = 0.0, save_trajectories = False):
        # Construct necessary attributes from parent class
        super().__init__(model, particle_count, save_trajectories)

        # store process and measurement noises
        self.process_noise_cov = self.model.hidden_state.sigma
        self.measurement_noise_cov = self.model.observation.sigma

        # parameters for the filter
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.aug_dimension = self.dimension + np.shape(self.process_noise_cov)[0] + np.shape(self.process_noise_cov)[0]
        self.lam = (self.alpha**2)*(self.aug_dimension + self.kappa) - self.aug_dimension

        # memory allocation for sigma points and weights
        self.sigma_pts = np.zeros((2*self.aug_dimension + 1, self.aug_dimension))
        self.sigma_weights_m = (0.5/(self.dimension + self.lam))*np.ones(2*self.aug_dimension + 1)
        self.sigma_weights_m[0] *= (2*self.lam)
        self.sigma_weights_c = np.copy(self.sigma_weights_m)
        self.sigma_weights_c[0] = self.sigma_weights_m[0] + (1.0 - self.alpha**2 + self.beta)

        #print("\n\nweights_c\n=====================\n{}\n".format(self.sigma_weights_c))
        #print("\n\nweights_m\n=====================\n{}\n".format(self.sigma_weights_m))


    def compute_sigma_pts(self):

        # construct augmented mean and covariance
        aug_mean  = np.concatenate((self.importance_mean, np.zeros(np.shape(self.process_noise_cov)[0]), np.zeros(np.shape(self.process_noise_cov)[0])))
        aug_cov = scipy.linalg.block_diag(self.importance_cov, self.process_noise_cov, self.measurement_noise_cov)

        # compute sigma points
        #print("\n\naug_cov\n=====================\n{}\n".format((self.aug_dimension + self.lam)*aug_cov))
        root_matrix = scipy.linalg.sqrtm((self.aug_dimension + self.lam)*aug_cov)
        self.sigma_pts[0] = aug_mean
        for i, column in enumerate(root_matrix):
            #print(aug_mean, column)
            self.sigma_pts[2*i + 1] = aug_mean + column
            self.sigma_pts[2*(i + 1)] = aug_mean - column

    def compute_weights(self, observation):
        if self.current_time > 0:
            # Compute chi and gamma
            chi = np.zeros((2*self.aug_dimension + 1, self.dimension))
            gamma = np.zeros((2*self.aug_dimension + 1, self.dimension))

            for i, pt in enumerate(self.sigma_pts):
                x = pt[ : self.dimension]
                process_noise = pt[self.dimension : self.dimension + np.shape(self.process_noise_cov)[0]]
                measurement_noise = pt[self.dimension + np.shape(self.process_noise_cov)[0] : ]
                chi[i] = self.model.hidden_state.func(self.current_time, x, process_noise)
                gamma[i] = self.model.observation.func(self.current_time, chi[i], measurement_noise)

            mean_chi = np.dot(self.sigma_weights_m, chi)
            mean_gamma = np.dot(self.sigma_weights_m, gamma)

            # Compute P_xx, P_xy, P_yy
            P_xx = np.zeros((self.dimension, self.dimension))
            P_yy = np.zeros((self.dimension, self.dimension))
            P_xy = np.zeros((self.dimension, self.dimension))

            for i, ch in enumerate(chi):
                vec1 = ch - mean_chi
                P_xx += self.sigma_weights_c[i] * np.outer(vec1, vec1)

                vec2 = gamma[i] - mean_gamma
                P_yy += self.sigma_weights_c[i] * np.outer(vec2, vec2)

                P_xy += self.sigma_weights_c[i] * np.outer(vec1, vec2)

            # for debugging
            #print("\n\nP_xx\n=====================\n{}\n".format(P_xx))
            #print("\n\nP_xx eigenvalues\n=====================\n{}\n".format(np.linalg.eig(P_xx)[0]))

            #print("\n\nP_xy\n=====================\n{}\n".format(P_xy))
            #print("\n\nP_xy eigenvalues\n=====================\n{}\n".format(np.linalg.eig(P_xy)[0]))

            #print("\n\nP_yy\n=====================\n{}\n".format(P_yy))
            #print("\n\nP_yy eigenvalues\n=====================\n{}\n".format(np.linalg.eig(P_yy)[0]))

            # Compute Kalman gain and importance mean and variance
            K = np.dot(P_xy, np.linalg.inv(P_yy))
            self.importance_mean  = mean_chi + np.dot(K, observation - mean_gamma)
            self.importance_cov = P_xx - np.linalg.multi_dot([K, P_yy, K.T])

            # print("\n\ncovariance\n=====================\n{}\n".format(cov))
            # print("\n\ncovariance eigenvalues\n=====================\n{}\n".format(np.linalg.eig(cov)[0]))

            # Sample new particles and compute weights
            new_particles = np.random.multivariate_normal(self.importance_mean, self.importance_cov, size = self.particle_count)
            for i, w in enumerate(self.weights):
                prob1 = self.model.hidden_state.conditional_pdf(self.current_time, new_particles[i], self.particles[i])
                prob2 = self.model.observation.conditional_pdf(self.current_time, observation, new_particles[i])
                prob3 = scipy.stats.multivariate_normal.pdf(new_particles[i], mean = self.importance_mean, cov = self.importance_cov)
                #print(prob1, prob2, prob3)
                self.weights[i] *= (prob1*prob2/prob3)
            self.prev_particles = self.particles
            self.particles = new_particles
        else:
            self.particles = self.model.hidden_state.sims[0].generate(self.particle_count)
            self.prev_particles = np.copy(self.particles)
            self.weights = np.array([self.model.observation.conditional_pdf(self.current_time, observation, x) for x in self.particles])
            #print("\n\nweights\n=====================\n{}\n".format(self.weights))

        # normalize weights
        #print('current time = {} and weights = {}'.format(self.current_time, self.weights[:10]))
        self.weights /= self.weights.sum()

        # compute mean and variance of the particles
        self.importance_mean = np.average(self.particles, weights = self.weights, axis = 0)
        self.importance_cov = np.zeros((self.dimension, self.dimension))
        for i, x in enumerate(self.particles):
            x_ = x - self.importance_mean
            self.importance_cov += self.weights[i]*np.outer(x_, x_)


    def mcmc(self, observation):
        new_particles = []
        for i, x in enumerate(self.particles):
            new_particles.append(x)
            prob1 = self.model.hidden_state.conditional_pdf(self.current_time, x, self.prev_particles[i])
            prob2 = self.model.observation.conditional_pdf(self.current_time, observation, x)
            prob3 = scipy.stats.multivariate_normal.pdf(x, mean = self.importance_mean, cov = self.importance_cov)
            q = prob1*prob2/prob3

            attempts = 0
            while True and attempts < 10:
                attempts += 1
                #print("\n\nimp_mean\n=====================\n{} {} {}\n".format(self.importance_mean, self.current_time, i))
                #print("\n\nimp_cov\n=====================\n{}\n".format(self.importance_cov))
                sample  = np.random.multivariate_normal(mean = self.importance_mean, cov = self.importance_cov)
                prob1 = self.model.hidden_state.conditional_pdf(self.current_time, sample, self.prev_particles[i])
                prob2 = self.model.observation.conditional_pdf(self.current_time, observation, sample)
                prob3 = scipy.stats.multivariate_normal.pdf(sample, mean = self.importance_mean, cov = self.importance_cov)
                p = prob1*prob2/prob3
                if np.random.random() <= min((1.0, p/q)):
                    new_particles[i] = sample
                    #self.weights[i] = p
                    break

        self.particles = np.array(new_particles)
        #self.weights /= self.weights.sum()

    def update(self, observations, threshold_factor = 0.1, method = 'mean', mcmc = False):
        """
        Description:
            Updates using all the obeservations using self.compute_weights and self.resample
        Args:
            observations: list/np.array of observations to pass to self.compute_weights
            threshold_factor: fraction of self.particle_count which the effective particle count has to surpass in order to stop resampling
        Returns:
            self.weights
        """
        self.current_time = 0
        self.observed_path = observations
        for observation in self.observed_path:
            self.compute_weights(observation = observation)
            resampled = self.resample(threshold_factor = threshold_factor)
            if resampled is True and mcmc is True:
                self.mcmc(observation = observation)
            if method is not None:
                self.compute_trajectory(method = method)
            self.compute_sigma_pts()
            self.current_time += 1
        return self.weights

















































class QuadraticImplicitPF(ParticleFilter):
    """
    Description:
        Defines an implicit particle filter that uses quadratic approximation of log of p(y|x)
    Parent class:
        ParticleFilter
    """
    def __init__(self, model, particle_count, grad, hessian, cholesky_factor_invT, save_trajectories = False):
        super().__init__(model = model, particle_count = particle_count, save_trajectories = save_trajectories)
        self.grad = grad # gradient of F_k, it's a function of form f(x, y, x_0)
        self.hessian = hessian # hessian of F_k, it's a function of form f(x, y, x_0)
        self.cholesky_factor_invT = cholesky_factor_invT # inverse transpose of Cholesky factor of H, it's a function of form f(x, y, x_0)
        self.std_mean = [0]*self.dimension
        self.std_cov = np.identity(self.dimension)
        # figure out covariances to define F_k = negative log of product of conditional pdfs
        self.dynamic_cov_inv = np.linalg.inv(self.model.hidden_state.error_cov)
        self.measurement_cov_inv = np.linalg.inv(self.model.observation.error_cov)

        # define F = negative log of product of conditional pdfs
        def F_k(x, y, x_0):
            a = x - self.model.hidden_state.f(x_0)
            b = y - self.model.observation.f(x)
            return 0.5*(np.dot(a.T, np.dot(self.dynamic_cov_inv, a)) + np.dot(b.T, np.dot(self.measurement_cov_inv, b)))

        self.F = F_k

    def compute_weights(self, observation):
        """
        Description:
            Updates weights according to the last observation
        Args:
            observation: an observation of dimension = self.dimension
        Returns:
            self.weights
        """
        if self.current_time < 1:
            # create a new dimension to add to the particles
            self.particles = self.model.hidden_state.sims[self.current_time].generate(self.particle_count)
            # compute new weights
            for i in range(self.particle_count):
                #prob1 = self.model.hidden_state.conditional_pdf(self.current_time, new_particles[i], self.particles[i])
                prob2 = self.model.observation.conditional_pdf(self.current_time, observation, self.particles[i])
                #prob3 = self.importance_pdf(new_particles[i], self.particles[i])
                self.weights[i] *= prob2
        else:
            #xi = np.random.multivariate_normal(self.std_mean, self.std_cov)
            for k in range(self.particle_count):
                # create F_k, its grad and hessian for minimization
                F_k = lambda x: self.F(x, observation, self.particles[k])
                hessian = lambda x: self.hessian(x, observation, self.particles[k])
                grad = lambda x: self.grad(x, observation, self.particles[k])

                # minimize F_k
                #res = opt.minimize(F_k, self.particles[k], method = 'BFGS', jac = grad)
                #mu, phi_k= res.x, res.fun
                mu =  0.5*(self.particles[k]+observation)
                phi_k = F_k(mu)
                #print("diff = {}".format(res.x - 0.5*(self.particles[k]+observation)) )
                # compute position of k-th particle
                xi = np.random.multivariate_normal(self.std_mean, self.std_cov)
                position = mu + np.dot(self.cholesky_factor_invT(phi_k, observation, self.particles[k]), xi)

                # compute weight of k-th particle
                a = position - mu
                #F_0 = phi_k + 0.5*np.dot(a.T, np.dot(hessian(mu), a))
                self.weights[k] *= np.exp(0.5*np.dot(xi, xi)-F_k(position))
                self.particles[k] = position
        #print('w={}'.format(self.weights[0]))
        # normalize weights
        self.weights /= self.weights.sum()

        #print(self.weights)
        #print(self.particles)
        if self.save_trajectories:
            self.trajectories = np.append(self.trajectories, [self.particles], axis = 0)

        self.current_time += 1
        #print('w_max = {}'.format(np.max(self.weights)))
        return self.weights



class RandomQuadraticIPF(ParticleFilter):
    """
    Description:
        Defines an implicit particle filter that uses quadratic approximation of log of p(y|x)
    Parent class:
        ParticleFilter
    """
    def __init__(self, model, particle_count, grad, minimum, save_trajectories = False):
        super().__init__(model = model, particle_count = particle_count, save_trajectories = save_trajectories)
        self.grad = grad # gradient of F_k, it's a function of form f(x, y, x_0)
        self.minimum = minimum # minimum of F_k, it's a function of form f(y, x_0)
        self.std_mean = [0]*self.dimension
        self.std_cov = np.identity(self.dimension)

        # figure out covariances to define F_k = negative log of product of conditional pdfs
        self.dynamic_cov_inv = np.linalg.inv(self.model.hidden_state.error_cov)
        self.measurement_cov_inv = np.linalg.inv(self.model.observation.error_cov)

        # define F = negative log of product of conditional pdfs
        def F_k(x, y, x_0):
            a = x - self.model.hidden_state.f(x_0)
            b = y - self.model.observation.f(x)
            return 0.5*(np.linalg.multi_dot([a, self.dynamic_cov_inv, a]) + np.linalg.multi_dot([b, self.measurement_cov_inv, b]))

        self.F = F_k

    def compute_weights(self, observation):
        """
        Description:
            Updates weights according to the last observation
        Args:
            observation: an observation of dimension = self.dimension
        Returns:
            self.weights
        """
        if self.current_time < 1:
            # create a new dimension to add to the particles
            self.particles = self.model.hidden_state.sims[self.current_time].generate(self.particle_count)
            # compute new weights
            for i in range(self.particle_count):
                #prob1 = self.model.hidden_state.conditional_pdf(self.current_time, new_particles[i], self.particles[i])
                prob2 = self.model.observation.conditional_pdf(self.current_time, observation, self.particles[i])
                #prob3 = self.importance_pdf(new_particles[i], self.particles[i])
                self.weights[i] *= prob2
        else:
            xi = np.random.multivariate_normal(self.std_mean, self.std_cov)
            for k in range(self.particle_count):
                # create F_k, its grad and hessian for minimization
                F_k = lambda x: self.F(x, observation, self.particles[k])

                # figure out phi_k and mu_k
                mu_k =  self.minimum(observation, self.particles[k])
                phi_k = F_k(mu_k)

                # create the non-linear equation
                #xi = np.random.multivariate_normal(self.std_mean, self.std_cov)
                rho = np.dot(xi, xi)
                eta = xi/np.sqrt(rho)
                f = lambda lam: F_k(mu_k + lam*eta) - phi_k - 0.5*rho

                # create the derivatives of F_k, f
                grad_F_k = lambda x: self.grad(x, observation, self.particles[k])
                fprime = lambda lam: [np.dot(grad_F_k(mu_k + lam*eta), eta)]

                # solve for current particle position and compute Jacobian
                lam = opt.fsolve(f, 0.01, fprime = fprime)[0]
                # print("{}----------{}".format(lam, f(lam)))
                J = lam**(self.dimension-1)*rho**(1-0.5*self.dimension)/fprime(lam)
                self.particles[k] = mu_k + lam*eta #** don't shift this line up

                # compute weight of k-th particle
                self.weights[k] *= np.exp(-phi_k)*abs(J)
        #print('w={}'.format(self.weights[0]))
        # normalize weights
        self.weights /= self.weights.sum()

        #print(self.weights)
        #print(self.particles)
        if self.save_trajectories:
            self.trajectories = np.append(self.trajectories, [self.particles], axis = 0)

        self.current_time += 1
        #print('w_max = {}'.format(np.max(self.weights)))
        return self.weights
