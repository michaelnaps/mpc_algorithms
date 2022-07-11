import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/test/models/.');

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

def plotIDQPTrackingResults(sim_time, tpm, savefile=0):
    T = tpm[0];
    sim_dt = T[1] - T[0];
    Nt = int(sim_time/sim_dt);
    T = T[:Nt];

    objA = np.transpose(tpm[3][:Nt]);
    objD = np.transpose(tpm[4][:Nt]);

    objFig, objPlot = plt.subplots(4,1,constrained_layout=True);

    objPlot[0].plot(T, objD[0], label='Desired', color="yellowgreen", linestyle='dashed', linewidth=2.5);
    objPlot[0].plot(T, objA[0], label='Actual', color="#1f77b4", linewidth=2.5);
    objPlot[0].set_title('COM x-Position Tracking', fontsize=14);
    objPlot[0].set_ylabel('$x_{c}\\ [m]$', fontsize=12);
    objPlot[0].legend(prop={"size":12});
    objPlot[0].grid();

    objPlot[1].plot(T[1:], objD[1][1:], label='Desired', color="yellowgreen", linestyle='dashed', linewidth=2.5);
    objPlot[1].plot(T[1:], objA[1][1:], label='Actual', color="#1f77b4", linewidth=2.5);
    objPlot[1].set_title('COM z-Position Tracking', fontsize=14);
    objPlot[1].set_ylabel('$h_{c}\\ [m]$', fontsize=12);
    objPlot[1].grid();

    objPlot[2].plot(T[1:], objD[2][1:], label='Desired', color="yellowgreen", linestyle='dashed', linewidth=2.5);
    objPlot[2].plot(T, [0] + objA[2], label='Actual', color="#1f77b4", linewidth=2.5);
    objPlot[2].set_title('Torso Orientation Tracking', fontsize=14);
    objPlot[2].set_ylabel('$q_{c}\\ [rad]$', fontsize=12);
    objPlot[2].set_xlabel('Time $[s]$', fontsize=12);
    objPlot[2].grid();

    objPlot[3].plot(T, objA[3], label='Actual', color="#1f77b4", linewidth=2.5);
    objPlot[3].plot(T, objD[3], label='Desired', color="yellowgreen", linestyle='dashed', linewidth=2.5);
    objPlot[3].set_title('COM Angular Momentum Tracking', fontsize=14);
    objPlot[3].set_ylabel('$L_{c}\\ [\\frac{kg\\ m^{2}}{s}]$', fontsize=12);
    objPlot[3].grid();

    if savefile != 0:
        objFig.set_size_inches(9,7.5);
        objFig.savefig(savefile, dpi=600);
        print("ID-QP objctive plot saved.\nFile path:", savefile);

    return (Nt, objA, objD, objPlot);

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

    labels = ['random'];
    Nl = len(labels);
    alip_results = {labels[i]: None for i in range(Nl)};
    tpm_results = {labels[i]: None for i in range(Nl)};

    for i in range(Nl):
        alip_results[labels[i]] = loadResults_alip('results/resultsALIP_' + labels[i] + '.pickle');
        tpm_results[labels[i]] = loadResults_tpm('results/resultsTPM_' + labels[i] + '.pickle');

    plot_time = tpm_results[labels[0]][0][-1];  sim_dt = 0.0005;
    Nt = int(plot_time/sim_dt);

    # random plots ------------------------------------------------------------------------------------------------------
    alip = alip_results['random'];
    tpm  = tpm_results['random'];

    plot_time = 15.0;
    sim_dt = tpm[0][1] - tpm[0][0];
    Nt = int(plot_time/sim_dt);

    savefile = "/home/michaelnaps/mpc_thesis/LaTex/figures/idqp_trend_random.png";
    (Nt, objA, objD, objPlot) = plotIDQPTrackingResults(15, tpm, savefile);

    (stateFig, statePlot) = plotStates_tpm(tpm[0][:Nt], tpm[1][:Nt]);
    stateFig.set_size_inches(8,6);
    # stateFig.savefig("/home/michaelnaps/mpc_thesis/LaTex/figures/state_trend_random.png", dpi=600);

    (inputFig, inputPlot) = plotInputs_tpm(tpm[0][:Nt], tpm[2][:Nt]);
    inputFig.set_size_inches(4,8);
    inputFig.savefig("input_trend_random.png", dpi=600);

    plt.show(block=0);

    input("Press enter to close random plots...");
    plt.close('all');
