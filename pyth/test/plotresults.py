import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import mpc
from statespace_alip import *
from statespace_tpm import *
import matplotlib.pyplot as plt

import gym
# from gym.wrappers import Monitor
import os
import numpy as np
import time
import argparse
import pickle
from datetime import datetime

from gym.envs.registration import registry, register, make, spec

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

    # alip_results = loadResults_alip("resultsALIP.pickle");
    # tpm_results  = loadResults_tpm("resultsTPM.pickle");

    plot_time = 60.0;  sim_dt = 0.0005;
    Nt = int(plot_time/sim_dt);

    ans = input("\nSee ALIP plots? [y/n] ");
    if (ans == 'y'):
        reportResults_alip(alip_results);

        fig, alipTrackingPlot = plt.subplots();

        s_actual = np.transpose(tpm_results[3])[3];
        s_desired = np.transpose(tpm_results[4])[3];

        alipTrackingPlot.plot(tpm_results[0][:Nt], s_actual[:Nt], label="Actual");
        alipTrackingPlot.plot(alip_results[0][:Nt], s_desired[:Nt], label="Desired");
        alipTrackingPlot.set_title("Lc Tracking")
        alipTrackingPlot.legend();
        plt.show(block=0);

        input("Press enter to close plots...");
        plt.close('all');

    ans = input("\nSee TPM plots? [y/n] ");
    if (ans == 'y'):
        statePlot = plotStates_tpm(tpm_results[0][:Nt], tpm_results[1][:Nt]);
        inputPlot = plotInputs_tpm(tpm_results[0][:Nt], tpm_results[2][:Nt]);
        plt.show(block=0);

        input("Press enter to close plots...");
        plt.close('all');

    ans = input("\nSee TPM animation? [y/n] ");
    if (ans == 'y'):
        anim_dt = 0.01;
        anim_Nt = int(plot_time/anim_dt + 1);
        jump = int(anim_dt/sim_dt);
        T = [i*anim_dt for i in range(anim_Nt)];
        q = tpm_results[1][:Nt+1:jump];

        print(len(T));
        print(len(q));

        animation_tpm(T, q, inputs_tpm);

        input("Press enter to exit program...");
