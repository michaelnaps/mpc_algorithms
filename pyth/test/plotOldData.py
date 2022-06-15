import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import nno
from statespace_alip import *
from statespace_3link import *
import matplotlib.pyplot as plt

class Inputs3link:
    def __init__(self):
        self.num_inputs           = 3;
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0, 0, 0];
        self.joint_masses         = [5, 5, 30];
        self.link_lengths         = [0.5, 0.5, 0.6];

mpc_results = loadResults_alip("pickle_data/completedRun_p2_k2.pickle");

T = mpc_results[0];
q = mpc_results[1];

inputs = Inputs3link();

q_example = [[0.357571103645510, 2.426450446298773, -1.213225223149386, 0, 0, 0], [0.357571103645510, 2.426450446298773, -1.213225223149386, 0, 0, 0]]

animation_3link(T[-2:], q_example, inputs, 1);
