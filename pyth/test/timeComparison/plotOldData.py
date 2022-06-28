import sys

sys.path.insert(0, '../../.');
sys.path.insert(0, '../models');

import mpc
from statespace_alip import *
import matplotlib.pyplot as plt

class InputsALIP:
    def __init__(self, prev_input):
        self.gravity_acc          = -9.81;
        self.joint_masses         = [80];
        self.link_lengths         = [2.0];
        self.CP_maxdistance       = 0.1;
        self.input_bounds         = [50];
        self.prev_input           = prev_input;

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

    PH_min = 1;  PH_max = 20;
    rd_num = 50;
    PH = [i for i in range(PH_min, PH_max+1)];
    nno_runtime = [0 for i in range(PH_min, PH_max+1)];
    ngd_runtime = [0 for i in range(PH_min, PH_max+1)];
    nno_itertime = [0 for i in range(PH_min, PH_max+1)];
    ngd_itertime = [0 for i in range(PH_min, PH_max+1)];
    nno_meanitercount = [0 for i in range(PH_min, PH_max+1)];
    ngd_meanitercount = [0 for i in range(PH_min, PH_max+1)];
    nno_brklist = [];  ngd_brklist = [];

    print("PH:", PH);

    for i in range(PH_min-1,PH_max):
        nno_runtimecount = 0;
        ngd_runtimecount = 0;

        for j in range(rd_num):
            nno_results = loadResults_alip("data/nnoResults_p" + str(PH[i]) + "_t" + str(j+1) + ".pickle");
            ngd_results = loadResults_alip("data/ngdResults_p" + str(PH[i]) + "_t" + str(j+1) + ".pickle");

            ngd_Nt = len(ngd_results[0]);
            nno_Nt = len(nno_results[0]);
            nno_check = 1;
            ngd_check = 1;
            nno_itercount = 0;
            ngd_itercount = 0;

            nno_check = np.sum([(nno_results[5][k] == -2) for k in range(nno_Nt)]) < 1;
            ngd_check = np.sum([(ngd_results[5][k] == -2) for k in range(ngd_Nt)]) < 1;

            if nno_check:
                nno_runtime[i] += np.mean(nno_results[6]);
                nno_runtimecount += 1;

                for k in range(nno_Nt):
                    if (nno_results[4][k] > 0):
                        nno_itertime[i] += nno_results[6][k];
                        nno_itercount += nno_results[4][k];

                nno_itertime[i] /= nno_itercount;
                nno_meanitercount[i] = nno_itercount/nno_Nt;

            if ngd_check:
                ngd_runtime[i] += np.mean(ngd_results[6]);
                ngd_runtimecount += 1;

                for k in range(ngd_Nt):
                    if (ngd_results[4][k] > 0):
                        ngd_itertime[i] += ngd_results[6][k];
                        ngd_itercount += ngd_results[4][k];

                ngd_itertime[i] /= ngd_itercount;
                ngd_meanitercount[i] = ngd_itercount/ngd_Nt;

            nno_brklist += nno_results[5];
            ngd_brklist += ngd_results[5];

        nno_runtime[i] = nno_runtime[i]/nno_runtimecount;
        ngd_runtime[i] = ngd_runtime[i]/ngd_runtimecount;

    fig, runTimeComparisonPlot = plt.subplots(1,3);

    runTimeComparisonPlot[0].bar(PH, nno_meanitercount, color="#1f77b4", label="NNO", zorder=10);
    runTimeComparisonPlot[0].bar(PH, ngd_meanitercount, color="yellowgreen", label="NGD", zorder=0);
    runTimeComparisonPlot[0].set_title("Number of Iterations");
    runTimeComparisonPlot[0].set_ylabel("Iteration Count [n]");
    runTimeComparisonPlot[0].legend();
    runTimeComparisonPlot[0].grid();

    runTimeComparisonPlot[1].plot(PH, nno_itertime, color="#1f77b4");
    runTimeComparisonPlot[1].plot(PH, ngd_itertime, color="yellowgreen");
    runTimeComparisonPlot[1].set_title("Iteration Time");
    runTimeComparisonPlot[1].set_ylabel("Time [ms]");
    runTimeComparisonPlot[1].set_xlabel("Prediction Horizon Length")
    runTimeComparisonPlot[1].grid();

    runTimeComparisonPlot[2].plot(PH, nno_runtime, color="#1f77b4");
    runTimeComparisonPlot[2].plot(PH, ngd_runtime, color="yellowgreen");
    runTimeComparisonPlot[2].set_title("MPC Runtime Time");
    runTimeComparisonPlot[2].set_ylabel("Time [ms]");
    runTimeComparisonPlot[2].grid();

    nno_brkFreqPlot = plotBrkFreq_alip(nno_brklist);
    nno_brkFreqPlot.set_title("NNO");
    ngd_brkFreqPlot = plotBrkFreq_alip(ngd_brklist);
    ngd_brkFreqPlot.set_title("NGD");

    plt.show(block=0);

    input("Press enter to close plots...");
    plt.close('all');
