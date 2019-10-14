import filter as fl
import numpy as np

size = 100
signal =  np.array([(t, t**2, t**3, t**4) for t in range(size)])
noise = np.random.normal(0.0, 1, size = (size, len(signal[0])))
observation = signal + noise
flt = fl.Filter(signal)
flt.plot_signals(signals = [signal, observation], labels = ['actual', 'observed'], coords_to_plot = [2, 1])
