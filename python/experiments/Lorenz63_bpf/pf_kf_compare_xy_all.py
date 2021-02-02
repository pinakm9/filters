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

pf_experiments = [0] * 5
pf_experiments[0] = 'xy_np_100_rn_0.05_pc_1.0_oc_0.1_og_0.2_s_0.0#0/bpf_assimilation.h5'
pf_experiments[1] = 'xy_np_200_rn_0.01_pc_1.0_oc_0.1_og_0.2_s_0.0#0/bpf_assimilation.h5'
pf_experiments[2] = 'xy_np_100_rn_0.05_pc_0.01_oc_0.1_og_0.2_s_2.0#0/bpf_assimilation.h5'
pf_experiments[3] = 'xy_np_800_rn_0.05_pc_0.01_oc_0.1_og_0.2_s_3.0#0/bpf_assimilation.h5'
pf_experiments[4] = 'xy_np_100_rn_0.05_pc_0.1_oc_0.1_og_0.2_s_3.0#0/bpf_assimilation.h5'

kf_experiments = ['xy_kf_analysis/{}.npy'.format(i) for i in range(5)]


for i in range(5):
    pf_kf_comp = cd.PFvsKF(pf_experiments[i], kf_experiments[i])
    pf_kf_comp.compare_with_resampling(num_samples=5000, k=100, noise_cov=0.01, saveas='xy_pf_vs_kf/{}'.format(i))
