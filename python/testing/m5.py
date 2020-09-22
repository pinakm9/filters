import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import model5 as mod
import numpy as np

model, a, b, x0 = mod.model(10)
pts = np.zeros((100, 2))
for i in range(100):
    pts[i] = model.hidden_state.sims[0].algorithm()

avg = np.average(pts, axis = 0)
print('x0 is {}, avg = {}, dist = {}'.format(x0, avg, np.linalg.norm(x0 - avg)))
ens = model.hidden_state.sims[0].generate(100)
avg = np.average(ens, axis = 0)
print('x0 is {}, avg = {}, dist = {}'.format(x0, avg, np.linalg.norm(x0 - avg)))
print(ens)
