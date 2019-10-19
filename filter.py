# Classes defining generic signal-processing filters
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import utility as ut
import simulate as sm

class Filter(object):
    """
    Description:
        A class defining a generic filter

    Attributes:
        signal:
        start_time:
        time_step:
        algorithm:
        processed:

    Methods:
        plot_signals:
    """
    def __init__(self, signal = [], start_time = 0.0, time_step = 1.0):
        """
        Args:
            algorithm = algorithm that processes the last obeservation
            start_time = time at first obeservation
            time_step = time step between consecutive observations
        """
        # assign basic attributes
        self.signal = signal
        self.start_time = start_time
        self.time_step = time_step
        self.algorithm = None
        self.processed = []

        # figure out the dimension of the problem
        if not np.isscalar(signal[0]):
            self.dimension = len(signal[0])
        else:
            self.dimension = 1

    @timer
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
            t = np.linspace(self.start_time, self.start_time + (len(self.signal)-1)*self.time_step, num = len(self.signal))
            for i, signal in enumerate(signals):
                signal = ut.Picker(signal).equidistant(objs_to_pick = max_pts)
                ax.plot(t, signal, label = labels[i], linestyle = line_styles[i])
            ax.xlabel('time({})'.format(time_unit))
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
    def __init__(self, model, particle_count, importance_pdf = None):
        """
        Args:
            model: a ModelPF object containing the dynamic and measurement models
            particle_count: number of particles to be used
        """
        self.model = model
        # draw self.particle_count samples from the prior distribution and reshape for hstacking later
        self.particles = np.reshape(model.hidden_state.sims[0].generate(particle_count), (particle_count, 1))
        self.weights = np.array([1.0/particle_count]*particle_count)
        self.particle_count = particle_count
        self.current_time = 0
        # if importance density is not provided we use the bootstrap filter
        if importance_pdf is None:
            importance_pdf = self.model.hidden_state.conditional_pdf
        self.importance_pdf = importance_pdf

    def update(self, observation):
        """
        Description:
            Updates weights according to the last observation
        """
        self.current_time += 1
        new_dimension = np.reshape(self.model.hidden_state.sim[self.current_time].generate(self.particle_count), (self.particle_count, 1))
        self.particles = np.hstack((self.particles, new_dimension))
        for i, w in enumerate(self.weights):
            prob1 = self.model.hidden_state.conditional_pdf(self.particles[i][-1], self.particles[i][-2])
            prob2 = self.model.observation.conditional_pdf(observation, self.particles[i][-1])
            prob3 = self.importance_pdf(self.particles[i][-1], self.particles[i][-2])
            self.weights[i] = w*prob1*prob2/prob3
        # normalize weights
        self.weights /= self.weights.sum()
