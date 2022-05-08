import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import nno
from statespace_lapm import *
import matplotlib.pyplot as plt

mpc = nno.load_results("pickle_data/lapmRun_p20_k1_t2.pickle");

T = mpc[0];
u = mpc[2];

mpcPlot = plotMPCComparison_lapm(T, u, 5);

plt.show();
