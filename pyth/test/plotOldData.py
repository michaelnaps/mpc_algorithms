import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import mpc
from statespace_alip import *
from statespace_tpm import *
import matplotlib.pyplot as plt

class InputsALIP:
    def __init__(self, prev_input):
        self.gravity_acc          = -9.81;
        self.joint_masses         = [40];
        self.link_lengths         = [0.95];
        self.CP_maxdistance       = 0.5;
        self.input_bounds         = [2];
        self.prev_input           = prev_input;

class InputsTPM:
    def __init__(self):
        self.num_inputs           = 3;
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0, 0, 0];
        self.joint_masses         = [5, 5, 30];
        self.link_lengths         = [0.5, 0.5, 0.6];

if (__name__ == "__main__"):
    inputs_alip = InputsALIP([0]);
    inputs_tpm = InputsTPM();

    alip_results = loadResults_alip("resultsALIP.pickle");
    tpm_results  = loadResults_tpm("resultsTPM.pickle");

    reportResults_alip(alip_results);


    ans = input("\nSee TPM plots? [y/n] ");
    if (ans == 'y'):
        statePlot = plotStates_tpm(tpm_results[0], tpm_results[1]);
        inputPlot = plotInputs_tpm(tpm_results[0], tpm_results[2]);
        plt.show(block=0);

        input("Press enter to close plots...");
        plt.close('all');
