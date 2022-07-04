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

    labels = ['base', 'heightchange', 'disturbance'];
    alip_results = {labels[i]: None for i in range(3)};
    tpm_results = {labels[i]: None for i in range(3)};

    for i in range(3):
        alip_results[labels[i]] = loadResults_alip('final_tests/resultsALIP_' + labels[i] + '.pickle');
        tpm_results[labels[i]] = loadResults_tpm('final_tests/resultsTPM_' + labels[i] + '.pickle');

    plot_time = tpm_results['base'][0][-1];  sim_dt = 0.0005;
    Nt = int(plot_time/sim_dt);

    # base test plots

    # height change test plots
    alip = alip_results['heightchange'];
    tpm  = tpm_results['heightchange'];

    objA = np.transpose(tpm[3]);
    objD = np.transpose(tpm[4]);

    fig, objPlot = plt.subplots(4,1,constrained_layout=True);

    objPlot[0].plot(tpm[0], objA[0], label='Actual', color="#1f77b4");
    objPlot[0].plot(tpm[0], objD[0], label='Desired', color="yellowgreen");
    objPlot[0].set_title('COM Location Tracking');
    objPlot[0].set_ylabel('Distance from Center [m]');
    objPlot[0].legend();  objPlot[0].grid();

    objPlot[1].plot(tpm[0][1:], objA[1][1:], label='Actual', color="#1f77b4");
    objPlot[1].plot(tpm[0][1:], objD[1][1:], label='Desired', color="yellowgreen");
    objPlot[1].set_title('Height Tracking');
    objPlot[1].set_ylabel('Height [m]');
    objPlot[1].set_xlabel('Time [s]');
    objPlot[1].grid();

    objPlot[2].plot(tpm[0][1:], objA[2][1:], label='Actual', color="#1f77b4");
    objPlot[2].plot(tpm[0][1:], objD[2][1:], label='Desired', color="yellowgreen");
    objPlot[2].set_title('Orientation Tracking');
    objPlot[2].set_ylabel('Angle of Torso Link [m]');
    objPlot[2].set_xlabel('Time [s]');
    objPlot[2].grid();

    plt.show(block=0);

    input("Press enter to close height change plots...");
    plt.close('all');

    # disturbance plots
