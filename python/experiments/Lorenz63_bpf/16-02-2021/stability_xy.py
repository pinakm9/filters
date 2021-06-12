import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import compare_dist as cd
import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(2021)

expr_folder = 'obs_gap_0.4'
assml_file = 'assimilation.h5'
experiments = [expr_folder + '/' + folder + '/' + assml_file for folder in os.listdir(expr_folder)]
num_exprs = len(experiments)

plt.figure(figsize = (8, 8))

for i in range(num_exprs):
    for j in range(i+1, num_exprs, 1):
        id1 = experiments[i].split('/')[1].split('_')[0]
        id2 = experiments[j].split('/')[1].split('_')[0]
        print('comparing {} vs {} ...'.format(id1, id2))
        pf_comp = cd.PFComparison(experiments[i], experiments[j])
        kl = pf_comp.compare_with_resampling(num_samples=1000, k=100, noise_cov=0.01, saveas=None)
        plt.plot(range(len(kl)), kl, label='{} vs {}'.format(id1, id2))
        
plt.title(expr_folder)
plt.plot(range(len(kl)), np.zeros(len(kl)), label='x-axis', color='black')
plt.legend()
plt.savefig('stability_{}/stability_{}.png'.format(expr_folder.split('_')[-1], len(kl)))
