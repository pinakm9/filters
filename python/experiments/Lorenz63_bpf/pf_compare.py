import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import compare_dist as cd

assml_file_1 = 'np_100_rn_005/bpf_assimilation.h5'
assml_file_2 = 'np_200_rn_001/bpf_assimilation.h5'

pf_comp = cd.PFComparison(assml_file_1, assml_file_2)
pf_comp.compare(num_samples=5000, k=10)
