import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import nno
from statespace_alip import *
import matplotlib.pyplot as plt

(inputs, mpc_var, mpc_results) = loadResults_alip("pickle_data/lapmRun_ngd_p10_k2_t2.pickle");

_ = reportResults_alip(inputs, mpc_var, mpc_results);
