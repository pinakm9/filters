# Classes defining generic signal-processing filters
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import utility as ut
import simulate as sm
import collections as cl

class Filter(object):
    """
    Description:
        A class defining a generic filter. Processes signals that are linear arrays and not matrices and tensors.

    Attributes:
        signal:
        start_time:
        time_step:
        algorithm:
        processed:

    Methods:
        plot_signals:
    """
    def __init__(self, signal = [], dimension = None, start_time = 0.0, time_step = 1.0):
        """
        Args:
            signal: signal to be processed
            start_time: time at first obeservation, default = 0.0
            time_step: time step between consecutive observations, default = 1.0
        """
        # assign basic attributes
        self.signal = signal
        self.start_time = start_time
        self.time_step = time_step
        self.algorithm = None
        self.processed = []

        # figure out the dimension of the problem
        if dimension is None:
            if len(np.shape(signal)) == 2:
                self.dimension = np.shape[1]
            else:
                self.dimension = 1
        else:
            self.dimension = dimension

    def plot_signals(self, signals, labels, line_styles = ['solid', 'dotted', 'dashed'],  max_pts = 100, fig_size = (7,6), time_unit = 'second', coords_to_plot = []):
        """
        Description:
            Plots observed and processed signals depending on the dimension of the problem

        Args:
            signals: signals to be plotted
            labels: identifiers for the signals
            line_styles: line styles for signals
            max_pts: Maximum number of points (default = 100) to be plotted for each signal
            fig_size: size of the plot as a tuple (unit of length as in matplotlib standard)
            time_unit: unit of time to be displayed in x-label for 1-dimensional problems
            coords_to_plot: list of coordinates to plot, default is [] for which all coordinates are plotted (together in case dimension < 4 and separately otherwise)

        Returns:
            figure and axes objects created (axes is a list of matplotlib axes in case coords_to_plot is not empty)
        """
        # prepare a figure
        fig = plt.figure(figsize = fig_size)

        # fix line_styles if its length is not adequate
        if len(signals) > len(line_styles):
            line_styles += ['solid']*(len(signals) - len(line_styles))

        # plot signals against time
        if self.dimension == 1:
            ax = fig.add_subplot(111)
            t = np.linspace(self.start_time, self.start_time + (len(signals[0])-1)*self.time_step, num = len(signals[0]))
            for i, signal in enumerate(signals):
                signal = ut.Picker(signal).equidistant(objs_to_pick = max_pts)
                ax.plot(t, signal, label = labels[i], linestyle = line_styles[i])
            ax.set(xlabel = 'time({})'.format(time_unit))
            ax.legend()

        # plot all coordinatres of the signals together, time is not shown
        elif self.dimension == 2 and coords_to_plot == []:
            ax = fig.add_subplot(111)
            for i, signal in enumerate(signals):
                signal = ut.Picker(signal).equidistant(objs_to_pick = max_pts)
                ax.plot(signal[:, 0], signal[:, 1], label = labels[i], linestyle = line_styles[i])
            ax.legend()

        # plot all coordinatres of the signals together, time is not shown
        elif self.dimension == 3 and coords_to_plot == []:
            ax = fig.add_subplot(111, projection='3d')
            for i, signal in enumerate(signals):
                signal = ut.Picker(signal).equidistant(objs_to_pick = max_pts)
                ax.plot3D(signal[:, 0], signal[:, 1], signal[:, 2], label = labels[i], linestyle = line_styles[i])
            ax.legend()

        # plot the required coordinates separately against time
        elif self.dimension > 3 or coords_to_plot != []:
            if coords_to_plot != []:
                coords_to_plot = list(range(self.dimension))
            ax, num_rows = [], len(coords_to_plot)
            t = np.linspace(self.start_time, self.start_time + (len(self.signal)-1)*self.time_step, num = len(self.signal))
            for i in range(num_rows):
                ax.append(fig.add_subplot(num_rows, 1, i+1))
                for j, signal in enumerate(signals):
                    signal = ut.Picker(signal[:, coords_to_plot[i]]).equidistant(objs_to_pick = max_pts)
                    ax[i].plot(t, signal, label = labels[j], linestyle = line_styles[j])
                    ax[i].set(ylabel = 'dimension {}'.format(coords_to_plot[i]))
                    ax[i].yaxis.set_label_position('right')
                ax[i].legend()
            fig.text(0.5, 0.05, 'time({})'.format(time_unit), ha='center', va='center')
        plt.show()
        return fig, ax


class ModelPF():
    """
    Description:
        A class for defining dynamic and measurement models for a particle filter.

    Attributes:
        hidden_state: a MarkovChain object simulating the hidden state
        observation: an SPConditional object simulating the observations
    """
    def __init__(self, size, prior, dynamic_algorithm, measurement_algorithm, dynamic_pdf, measurement_pdf):
        """
        Args:
            size: length of the MarkovChain self.hidden_state
            prior: a Simulation object for defining the MarkovChain self.hidden_state
            dynamic_algorithm: algorithm for defining self.hidden_state
            measurement_algorithm: algorithm for defining self.observation
            dynamic_pdf: p(x_k|x_(k-1)), x_k is the hidden state
            measurement_pdf: p(y_k|x_k), y_k is the observation
        """
        # create the Markov chain of hidden state X_t and observation Y_t
        self.hidden_state = sm.MarkovChain(size = size, prior = prior, algorithm = dynamic_algorithm, conditional_pdf = dynamic_pdf)
        self.observation = sm.SPConditional(conditions = self.hidden_state.sims, algorithm = measurement_algorithm, conditional_pdf = measurement_pdf)

class ParticleFilter(Filter):
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
    def __init__(self, model, particle_count, importance_pdf = None, start_time = 0.0, time_step = 1.0):
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
        self.particles = np.reshape(model.hidden_state.sims[0].generate(particle_count), (particle_count, 1))
        self.particle_count = particle_count
        self.weights = np.ones(particle_count)/particle_count
        self.current_time = 0
        # if importance density is not provided we use the bootstrap filter
        if importance_pdf is None:
            importance_pdf = self.model.hidden_state.conditional_pdf
        self.importance_pdf = importance_pdf
        super().__init__(dimension = self.model.hidden_state.sims[0].dimension, start_time = start_time, time_step = time_step)

    def compute_weights(self, observation):
        """
        Description:
            Updates weights according to the last observation

        Args:
            observation: an observation of dimension = self.dimension

        Returns:
            self.weights
        """
        # tick the internal timer
        self.current_time += 1
        if self.current_time >= self.model.hidden_state.size:
            return self.weights

        # create a new dimension to add to the particles
        new_dimension = np.reshape(self.model.hidden_state.sims[self.current_time].generate(self.particle_count), (self.particle_count, 1))
        self.particles = np.hstack((self.particles, new_dimension))

        # compute new weights
        for i, w in enumerate(self.weights):
            prob1 = self.model.hidden_state.conditional_pdf(self.particles[i][-1], self.particles[i][-2])
            prob2 = self.model.observation.conditional_pdf(observation, self.particles[i][-1])
            prob3 = self.importance_pdf(self.particles[i][-1], self.particles[i][-2])
            self.weights[i] = w*prob1*prob2/prob3

        # normalize weights
        self.weights /= self.weights.sum()

        return self.weights

    def resample(self, threshold_factor = 0.1):
        """
        Description:
            Performs resampling

        Args:
            threshold_factor: fraction of self.particle_count which the effective particle count has to surpass in order to stop resampling

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

    def update(self, observations, threshold_factor = 0.5):
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
        return self.weights

    def filtering_pdf(self, x, time):
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
            result += self.weights[i]*ut.delta(x, self.particles[i][time])
        return result

    def compute_hidden_state(self, method = 'mean'):
        """
        """
        self.hidden_state = np.zeros(self.model.hidden_state.size)

        if method == 'mode':
            # for each time find the most likely particle
            for k in range(self.model.hidden_state.size):
                self.hidden_state[k] = self.particles[np.array(list(map(lambda x: self.filtering_pdf(x, k), self.particles[:, k]))).argmax()][k]
        elif method == 'mean':
            for k in range(self.model.hidden_state.size):
                self.hidden_state[k] = np.average(a = list(map(lambda x: self.filtering_pdf(x, k), self.particles[:, k])), weights = self.weights)
        return self.hidden_state
