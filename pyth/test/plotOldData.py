import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import nno
from statespace_lapm import *
import matplotlib.pyplot as plt

(inputs, mpc_var, mpc_results) = loadResults_lapm("prevRun_data.pickle");

_ = reportResults_lapm(inputs, mpc_var, mpc_results);
