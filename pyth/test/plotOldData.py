import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import nno
from statespace_alip import *
import matplotlib.pyplot as plt

mpc_results = loadResults_alip("prevRun_data.pickle");

_ = reportResults_alip(mpc_results);
