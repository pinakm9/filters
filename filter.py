# Classes defining generic signal-processing filters
import numpy as np
import utility as ut
import simulate as sm
import collections as cl
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
    def __init__(self, model, particle_count, importance_pdf = None, start_time = 0.0, time_step = 1.0, save_trajectories = False):
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
        self.hidden_trajectory = np.empty((0, self.model.hidden_state.dimension))
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

        # create a new dimension to add to the particles
        new_particles = self.model.hidden_state.sims[self.current_time].generate(self.particle_count)

        # compute new weights
        for i, w in enumerate(self.weights):
            #prob1 = self.model.hidden_state.conditional_pdf(new_particles[i], self.particles[i])
            prob2 = self.model.observation.conditional_pdf(observation, new_particles[i])
            #prob3 = self.importance_pdf(new_particles[i], self.particles[i])
            self.weights[i] = w*prob2
        #print(self.weights.sum(), np.max(self.weights))
        # normalize weights
        self.weights /= self.weights.sum()

        self.particles = new_particles
        if self.save_trajectories:
            self.trajectories = np.append(self.trajectories, [self.particles], axis = 0)

        self.current_time += 1
        return self.weights

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
            self.particles = np.take(a = self.particles, indices = indices, axis = 0)
            self.weights = np.ones(self.particle_count)/self.particle_count

            # create weight map for faster computation
            index_map = dict(cl.Counter(indices))
            self.weight_map = np.zeros((len(index_map), 2))
            for i, (key, value) in enumerate(index_map.items()):
                self.weight_map[i] = [key, value*self.weights[0]]

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

    def compute_hidden_trajectory(self, method = 'mean'):
        """
        """
        if method == 'mode':
            # for each time find the most likely particle
            new_hidden_state = self.particles[np.array(list(map(self.filtering_pdf, self.particles))).argmax()]
            self.hidden_trajectory = np.append(self.hidden_trajectory, [new_hidden_state], axis = 0)
        elif method == 'mean':
            new_hidden_state = np.average(self.particles, weights = self.weights, axis = 0)
            self.hidden_trajectory = np.append(self.hidden_trajectory, [new_hidden_state], axis = 0)
        return self.hidden_trajectory

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
            self.compute_hidden_trajectory(method = method)
        return self.weights

    def weight_graph(self):
        """
        Plots a histogram of weights
        """
        pass

#class ImplicitParticleFilter(ParticleFilter):
