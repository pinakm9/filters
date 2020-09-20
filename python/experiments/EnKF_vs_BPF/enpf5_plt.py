# Takes size (or length of Markov chain or final time) of models as the command line argument
# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_path = Path(dirname(realpath(__file__)))
module_dir = str(script_path.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import plot
image_dir = str(script_path.parent.parent.parent) + '/images/EnKF_vs_BPF/'
plot.im2pdf(im_folder = image_dir + 'EnKF_vs_BPF_frames', im_prefix = 'frame_', num_im = 100, im_format = 'png', pdf_name = image_dir + 'EnKF_vs_BPF.pdf')
plot.im2pdf(im_folder = image_dir + 'EnKF_vs_APF_frames', im_prefix = 'frame_', num_im = 100, im_format = 'png', pdf_name = image_dir + 'EnKF_vs_APF0_small.pdf')
