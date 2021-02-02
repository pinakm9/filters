import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent.parent.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import compare_dist as cd
import numpy as np
import os

np.random.seed(2021)

expr_folder = 'obs_gap_0.8'
assml_file = 'assimilation.h5'
experiments = [expr_folder + '/' + folder + '/' + assml_file for folder in os.listdir(expr_folder)[:-2]]
num_exprs = len(experiments)

for i in range(num_exprs):
    for j in range(i+1, num_exprs, 1):
        id1 = experiments[i].split('/')[1].split('_')[0]
        id2 = experiments[j].split('/')[1].split('_')[0]
        print('comparing {} vs {} ...'.format(id1, id2))
        pf_comp = cd.PFComparison(experiments[i], experiments[j])
        pf_comp.compare_with_resampling(num_samples=5000, k=100, noise_cov=0.01, saveas='stability_{}/{}_vs_{}'\
                                .format(expr_folder.split('_')[-1], id1, id2))
