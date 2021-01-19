import tables
import numpy as np
# modify according to your own h5 assimilation file
assimilation_file = 'np_100/bpf_assimilation.h5'
hdf5 = tables.open_file(assimilation_file, 'r')
observation = np.array(hdf5.root.observation.read().tolist())
print(observation)
