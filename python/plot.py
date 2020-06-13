import numpy as np
import utility as ut
import  matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class SignalPlotter(object):
    """
    Description:
        A class for plotting signals. Signal is a timeseries whose every can be a scalar or a vector (matrices and higher rank tensors are not supported).

    Attributes:
        signal:
        start_time:
        time_step:
        algorithm:

    Methods:
        plot_signals:
    """
    def __init__(self, signals = [], dimension = None, start_time = 0.0, time_step = 1.0):
        """
        Args:
            signals: signals to be processed
            start_time: time at first obeservation, default = 0.0
            time_step: time step between consecutive observations, default = 1.0
        """
        # assign basic attributes
        self.signals = signals
        self.start_time = start_time
        self.time_step = time_step
        self.algorithm = None
        self.processed = []

        # figure out the dimension of the problem
        if dimension is None:
            if len(np.shape(signals[0])) == 2:
                self.dimension = np.shape(signals[0])[1]
            else:
                self.dimension = 1
        else:
            self.dimension = dimension

    def plot_signals(self, labels = [], styles = [{'linestyle':'solid'}, {'marker':'o'}, {'marker':'^'}],\
                    plt_fns = ['plot', 'scatter', 'scatter'],  colors = ['red', 'green', 'blue'],\
                    max_pts = 100, fig_size = (7,6), time_unit = 'second', coords_to_plot = [],\
                    show = False, save = False, file_path = None, title = None):
        """
        Description:
            Plots observed and processed signals depending on the dimension of the problem

        Args:
            labels: identifiers for the signals
            styles: line styles for signals
            max_pts: Maximum number of points (default = 100) to be plotted for each signal
            fig_size: size of the plot as a tuple (unit of length as in matplotlib standard)
            time_unit: unit of time to be displayed in x-label for 1-dimensional problems
            coords_to_plot: list of coordinates to plot, default is [] for which all coordinates are plotted (together in case dimension < 4 and separately otherwise)

        Returns:
            figure and axes objects created (axes is a list of matplotlib axes in case coords_to_plot is not empty)
        """
        # prepare a figure
        fig = plt.figure(figsize = fig_size)

        # fix styles, labels, plt_fns and colors if their lengths are not adequate
        if len(self.signals) > len(styles):
            styles += [{'marker': 'x'}]*(len(self.signals) - len(styles))
        if len(self.signals) > len(labels):
            labels += ['']*(len(self.signals) - len(labels))
        if len(self.signals) > len(plt_fns):
            plt_fns += ['scatter']*(len(self.signals) - len(plt_fns))
        if len(self.signals) > len(colors):
            colors += ['blue']*(len(self.signals) - len(colors))

        # plot self.signals against time
        if self.dimension == 1 and coords_to_plot == []:
            ax = fig.add_subplot(111)
            t = np.linspace(self.start_time, self.start_time + (len(self.signals[0])-1)*self.time_step, num = min(max_pts, len(self.signals[0])))
            for i, signal in enumerate(self.signals):
                signal = ut.Picker(signal).equidistant(objs_to_pick = max_pts)
                getattr(ax, plt_fns[i])(t, signal, label = labels[i], color = colors[i], **styles[i])
            ax.set(xlabel = 'time({})'.format(time_unit))
            ax.legend()

        # plot all coordinatres of the self.signals together, time is not shown
        elif self.dimension == 2 and coords_to_plot == []:
            ax = fig.add_subplot(111)
            for i, signal in enumerate(self.signals):
                signal = ut.Picker(signal).equidistant(objs_to_pick = max_pts)
                getattr(ax, plt_fns[i])(signal[:, 0], signal[:, 1], label = labels[i], color = colors[i], **styles[i])
            ax.legend()

        # plot all coordinatres of the self.signals together, time is not shown
        elif self.dimension == 3 and coords_to_plot == []:
            ax = fig.add_subplot(111, projection='3d')
            for i, signal in enumerate(self.signals):
                signal = ut.Picker(signal).equidistant(objs_to_pick = max_pts)
                getattr(ax, plt_fns[i])(signal[:, 0], signal[:, 1], signal[:, 2], label = labels[i], color = colors[i], **styles[i])
            ax.legend()

        # plot the required coordinates separately against time
        elif self.dimension > 3 or coords_to_plot != []:
            ax, num_rows = [], len(coords_to_plot)
            t = np.linspace(self.start_time, self.start_time + (len(self.signals[0])-1)*self.time_step, min(max_pts, len(self.signals[0])))
            for i in range(num_rows):
                ax.append(fig.add_subplot(num_rows, 1, i+1))
                for j, signal in enumerate(self.signals):
                    signal = ut.Picker(signal[:, coords_to_plot[i]]).equidistant(objs_to_pick = max_pts)
                    getattr(ax[i], plt_fns[j])(t, signal, label = labels[j], color = colors[j], **styles[j])
                    ax[i].set(ylabel = 'dimension {}'.format(coords_to_plot[i] + 1))
                    ax[i].yaxis.set_label_position('right')
                ax[i].legend()
            fig.text(0.5, 0.05, 'time({})'.format(time_unit), ha='center', va='center')

        if title is not None:
            plt.title(title)
        if show is True:
            plt.show()
        if save is True:
            if file_path is not None:
                plt.savefig(fname = file_path)
            else:
                print("file_path was not specified. So the image file was not saved.")
        return fig, ax
