# Classes defining generic signal-processing filters
import numpy as np
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
    def __init__(self, model, particle_count, importance_pdf = None, save_trajectories = False):
        """
        Args:
            model: a ModelPF object containing the dynamic and measurement models
            particle_count: number of particles to be used
            importance_pdf: importance pdf for the particle filter, it's a function of form p(x, condition) (argument names can be anything)
            start_time: time at first obeservation, default = 0.0
            time_step: time step between consecutive observations, default = 1.0
        """
        self.model = model
        # draw self.particle_count samples from the prior distribution and reshape for hstacking later
        self.particles = []
        self.particle_count = particle_count
        self.weights = np.ones(particle_count)/particle_count
        self.current_time = 0
        # if importance density is not provided we use the bootstrap filter
        if importance_pdf is None:
            self.importance_pdf = self.model.hidden_state.conditional_pdf
        else:
            self.importance_pdf = importance_pdf
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
            #prob1 = self.model.hidden_state.conditional_pdf(new_particles[i], self.particles[i])
            prob2 = self.model.observation.conditional_pdf(observation, self.particles[i])
            #prob3 = self.importance_pdf(new_particles[i], self.particles[i])
            self.weights[i] *= prob2
        # print(self.weights.sum(), np.max(self.weights))
        # normalize weights
            self.weights /= self.weights.sum()

        if self.save_trajectories:
            self.trajectories = np.append(self.trajectories, [self.particles], axis = 0)

        self.current_time += 1
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
            print("\n\n$$$$$$$$$$$$$$$$$$$$$ Num particles = {} $$$$$$$$$$$$$$$$$$$$\n\n".format(u))
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

    def update(self, observations, threshold_factor = 0.5, method = 'mean'):
        """
        Description:
            Updates using all the obeservations using self.compute_weights and self.resample
        Args:
            observations: list/np.array of observations to pass to self.compute_weights
            threshold_factor: fraction of self.particle_count which the effective particle count has to surpass in order to stop resampling
        Returns:
            self.weights
        """
        for observation in observations:
            self.compute_weights(observation = observation)
            self.resample(threshold_factor = threshold_factor)
            if method is not None:
                self.compute_trajectory(method = method)
        return self.weights

    def compute_error(self):
        """
        Description:
            Computes error in assimilation for a random path
        """
        error = self.true_trajectory - self.computed_trajectory
        return np.linalg.norm(error), np.mean(error, axis = 0), np.std(error, axis = 0)


    def ecdf(self, x):
        result = 0.0
        for i in range(self.particle_count):
            result += self.weights[i]*np.prod(x > self.particles[i])
        return result


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
                #prob1 = self.model.hidden_state.conditional_pdf(new_particles[i], self.particles[i])
                prob2 = self.model.observation.conditional_pdf(observation, self.particles[i])
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
                #prob1 = self.model.hidden_state.conditional_pdf(new_particles[i], self.particles[i])
                prob2 = self.model.observation.conditional_pdf(observation, self.particles[i])
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
