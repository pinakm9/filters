# Classes defining generic signal-processing filters
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import utility as ut

class Filter(object):
    """
    A class defining a generic filter
    algorithm = algorithm that processes the last obeservation
    start_time = time at first obeservation
    time_step = time step between consecutive observations
    """
    def __init__(self, signal = [], start_time = 0.0, time_step = 1.0):
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

    @ut.timer
    def plot_signals(self, signals, labels, line_styles = ['solid', 'dotted', 'dashed'],  max_pts = 100, fig_size = (7,6), time_unit = 'second', coords_to_plot = []):
        """
        Plots observed and processed signals depending on the dimension of the problem
        signals = signals to be plotted
        labels = identifiers for the signals
        line_styles = line styles for signals
        max_pts = Maximum number of points (default = 100) to be plotted for each signal
        fig_size = size of the plot as a tuple (unit of length as in matplotlib standard)
        time_unit = unit of time to be displayed in x-label for 1-dimensional problems
        coords_to_plot = list of coordinates to plot, default is [] for which all coordinates are plotted (together in case dimension < 4 and separately otherwise)
        Returns figure and axes objects created (axes is a list of matplotlib axes in case coords_to_plot is not empty)
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

"""
    def compare(self):
        ""
        Plots the observed and processed signals
        ""
        pass
"""
