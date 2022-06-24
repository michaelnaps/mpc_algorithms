import sys

sys.path.insert(0, '../../.');
sys.path.insert(0, '../models');

import mpc
from statespace_alip import *
from statespace_3link import *
import matplotlib.pyplot as plt

class InputsALIP:
    def __init__(self, prev_input):
        self.gravity_acc          = -9.81;
        self.joint_masses         = [80];
        self.link_lengths         = [2.0];
        self.CP_maxdistance       = 0.1;
        self.input_bounds         = [50];
        self.prev_input           = prev_input;

class Inputs3link:
    def __init__(self):
        self.num_inputs           = 3;
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0, 0, 0];
        self.joint_masses         = [5, 5, 30];
        self.link_lengths         = [0.5, 0.5, 0.6];

if (__name__ == "__main__"):
    inputs = InputsALIP([0]);

    """
    ngd_results = loadResults_alip("ngdResults.pickle");
    nno_results = loadResults_alip("nnoResults.pickle");

    # reportResults_alip(ngd_results, inputs);
    # reportResults_alip(nno_results, inputs);

    ngd_runtime = plotRunTime_alip(ngd_results[0], ngd_results[6], title=0);
    nno_runtime = plotRunTime_alip(nno_results[0], nno_results[6], title=0);
    plt.show(block=0);

    ngd_mean = np.mean(ngd_results[6]);
    nno_mean = np.mean(nno_results[6]);

    fig, runTimeComparison = plt.subplots();
    runTimeComparison.plot(ngd_results[0], ngd_results[6], label='NGD', color='#1f77b4');
    runTimeComparison.plot([ngd_results[0][0], ngd_results[0][-1]], [ngd_mean, ngd_mean], color='#1f77b4', linestyle='--');
    runTimeComparison.plot(nno_results[0], nno_results[6], label='NNO', color='#2ca02c');
    runTimeComparison.plot([nno_results[0][0], nno_results[0][-1]], [nno_mean, nno_mean], color='#2ca02c', linestyle='--');
    runTimeComparison.plot([0], [0], label="AVE.", color='k', linestyle='--');
    plt.legend();
    plt.grid();
    plt.show(block=0);

    input("Press enter to close run time plots...");
    plt.close('all');
    """

    PH_num = 30;
    PH = [(i + 1) for i in range(PH_num)];
    ngd_meanlist = [0 for i in range(PH_num)];
    nno_meanlist = [0 for i in range(PH_num)];

    for i in range(PH_num):
        ngd_results = loadResults_alip("./results/ngdResults_p" + str(PH[i]) + ".pickle");
        nno_results = loadResults_alip("./results/nnoResults_p" + str(PH[i]) + ".pickle");

        ngd_meanlist[i] = np.mean(ngd_results[6]);
        nno_meanlist[i] = np.mean(nno_results[6]);

    fig, runTimeAverageComparison = plt.subplots();
    runTimeAverageComparison.plot(PH, ngd_meanlist, label="NGD");
    runTimeAverageComparison.plot(PH, nno_meanlist, label="NNO");
    plt.legend();
    plt.grid();
    plt.show(block=0);

    input("Press enter to close run time plots...");
    plt.close('all');
