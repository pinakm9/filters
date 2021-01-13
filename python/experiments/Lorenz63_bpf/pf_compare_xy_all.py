import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import compare_dist as cd
import numpy as np
np.random.seed(2021)

experiments = [0] * 5
experiments[0] = 'xy_np_100_rn_0.05_pc_1.0_oc_0.1_og_0.2_s_0.0#1/bpf_assimilation.h5'
experiments[1] = 'xy_np_200_rn_0.01_pc_1.0_oc_0.1_og_0.2_s_0.0#1/bpf_assimilation.h5'
experiments[2] = 'xy_np_100_rn_0.05_pc_0.01_oc_0.1_og_0.2_s_2.0#1/bpf_assimilation.h5'
experiments[3] = 'xy_np_800_rn_0.05_pc_0.01_oc_0.1_og_0.2_s_3.0#1/bpf_assimilation.h5'
experiments[4] = 'xy_np_100_rn_0.05_pc_0.1_oc_0.1_og_0.2_s_3.0#1/bpf_assimilation.h5'

for i, e1 in enumerate(experiments):
    for j in range(i+1, len(experiments), 1):
        pf_comp = cd.PFComparison(e1, experiments[j])
        pf_comp.compare(num_samples=5000, k=100, saveas='xy_filter_stability/{}_vs_{}'.format(i, j))
