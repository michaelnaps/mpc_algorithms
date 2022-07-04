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

    labels = ['base', 'heightchange', 'disturbance', 'random'];
    Nl = len(labels);
    alip_results = {labels[i]: None for i in range(Nl)};
    tpm_results = {labels[i]: None for i in range(Nl)};

    for i in range(Nl):
        alip_results[labels[i]] = loadResults_alip('final_tests/resultsALIP_' + labels[i] + '.pickle');
        tpm_results[labels[i]] = loadResults_tpm('final_tests/resultsTPM_' + labels[i] + '.pickle');

    plot_time = tpm_results['base'][0][-1];  sim_dt = 0.0005;
    Nt = int(plot_time/sim_dt);

    # base test plots

    # height change test plots
    """

    alip = alip_results['heightchange'];
    tpm  = tpm_results['heightchange'];

    Nt = int(15/0.0005);
    T = tpm[0][:Nt];
    objA = np.transpose(tpm[3][:Nt]);
    objD = np.transpose(tpm[4][:Nt]);

    fig, objPlot = plt.subplots(3,1,constrained_layout=True);

    objPlot[0].plot(T, objA[0], label='Actual', color="#1f77b4", linewidth=2.5);
    objPlot[0].plot(T, objD[0], label='Desired', color="yellowgreen", linestyle='dashed', linewidth=2.5);
    objPlot[0].set_title('COM Location Tracking', fontsize=14);
    objPlot[0].set_ylabel('COM Deviation $[m]$', fontsize=14);
    objPlot[0].legend(prop={"size":12});
    objPlot[0].grid();

    objPlot[1].plot(T[1:], objA[1][1:], label='Actual', color="#1f77b4", linewidth=2.5);
    objPlot[1].plot(T[1:], objD[1][1:], label='Desired', color="yellowgreen", linestyle='dashed', linewidth=2.5);
    objPlot[1].set_title('COM Height Tracking', fontsize=14);
    objPlot[1].set_ylabel('Height $[m]$', fontsize=14);
    objPlot[1].set_xlabel('Time $[s]$');
    objPlot[1].grid();

    objPlot[2].plot(T, [0] + objA[2], label='Actual', color="#1f77b4", linewidth=2.5);
    objPlot[2].plot(T[1:], objD[2][1:], label='Desired', color="yellowgreen", linestyle='dashed', linewidth=2.5);
    objPlot[2].set_title('Orientation Tracking', fontsize=14);
    objPlot[2].set_ylabel('Angle of Torso Link $[m]$', fontsize=14);
    objPlot[2].set_xlabel('Time $[s]$');
    objPlot[2].grid();

    plt.show(block=0);

    input("Press enter to close height change plots...");
    plt.close('all');

    """

    # disturbance plots

    # random plots
    alip = alip_results['random'];
    tpm  = tpm_results['random'];

    Nt = int(15/0.0005);
    T = tpm[0][:Nt];
    objA = np.transpose(tpm[3][:Nt]);
    objD = np.transpose(tpm[4][:Nt]);

    objFig, objPlot = plt.subplots(3,1,constrained_layout=True);

    objPlot[0].plot(T, objA[0], label='Actual', color="#1f77b4", linewidth=2.5);
    objPlot[0].plot(T, objD[0], label='Desired', color="yellowgreen", linestyle='dashed', linewidth=2.5);
    objPlot[0].set_title('COM Location Tracking from ID-QP', fontsize=14);
    objPlot[0].set_ylabel('COM Deviation $[m]$', fontsize=14);
    objPlot[0].legend(prop={"size":12});
    objPlot[0].grid();

    objPlot[1].plot(T[1:], objA[1][1:], label='Actual', color="#1f77b4", linewidth=2.5);
    objPlot[1].plot(T[1:], objD[1][1:], label='Desired', color="yellowgreen", linestyle='dashed', linewidth=2.5);
    objPlot[1].set_title('COM Height Tracking from ID-QP', fontsize=14);
    objPlot[1].set_ylabel('Height $[m]$', fontsize=14);
    objPlot[1].set_xlabel('Time $[s]$');
    objPlot[1].grid();

    objPlot[2].plot(T, [0] + objA[2], label='Actual', color="#1f77b4", linewidth=2.5);
    objPlot[2].plot(T[1:], objD[2][1:], label='Desired', color="yellowgreen", linestyle='dashed', linewidth=2.5);
    objPlot[2].set_title('Orientation Tracking from ID-QP', fontsize=14);
    objPlot[2].set_ylabel('Angle of Torso Link $[m]$', fontsize=14);
    objPlot[2].set_xlabel('Time $[s]$');
    objPlot[2].grid();

    objFig.set_size_inches(9,7.5);
    objFig.savefig("/home/michaelnaps/mpc_thesis/LaTex/figures/state_trend_random.png", dpi=600);

    amFig, amPlot = plt.subplots();

    amPlot.plot(T, objA[3], label='Actual', color="#1f77b4", linewidth=2.5);
    amPlot.plot(T, objD[3], label='Desired', color="yellowgreen", linestyle='dashed', linewidth=2.5);
    amPlot.set_title('ID-QP Tracking of MPC Angular Momentum', fontsize=14);
    amPlot.set_ylabel('Angular Momentum $[\\frac{kg\\ m^{2}}{s}]$', fontsize=14);
    amPlot.legend(prop={"size":12});
    amPlot.grid();

    amFig.set_size_inches(10,5);
    amFig.savefig("/home/michaelnaps/mpc_thesis/LaTex/figures/angular_momentum_trend_random.png", dpi=600);

    plt.show(block=0);

    input("Press enter to close random initial position plots...");
    plt.close('all');

    print("x_c error:", np.abs(objD[0][-1]-objA[0][-1]));
    print("h_c error:", np.abs(objD[1][-1]-objA[1][-1]));
    print("q_c error:", np.abs(objD[2][-1]-objA[2][-1]));
