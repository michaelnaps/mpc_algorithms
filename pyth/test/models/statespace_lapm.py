import numpy as np
import matplotlib.pyplot as plt
import pickle

def statespace_lapm(q, u, inputs):
    # constant variables
    m = inputs.joint_masses[0];
    H = inputs.link_lengths[0];
    g = inputs.gravity_acc;

    # state variables
    x  = q[0];  Lc  = q[1];
    dx = q[2];  dLc = q[3];

    # input variables
    ua = u[0];  L = u[1];

    # solve for statespace
    ddq = [
        L/(m*H) - Lc/(m*H),
        ua + m*g*x
    ];

    return [dx, dLc, ddq[0], ddq[1]];

# LAPM animation function
def animation_lapm(T, q, inputs):
    # sim variables
    Nt = len(T);
    dt = T[1] - T[0];

    # constant parameters
    H = inputs.link_lengths[0];
    L = inputs.link_lengths[0];
    d = inputs.CP_maxdistance;
    m = inputs.joint_masses[0];

    # limits of plot axes
    yLimits = [-0.1, L+L/10];
    xLimits = [-L/3, L/3];

    for i in range(Nt):
        plt.clf();

        x  = q[i][0];
        th = np.arccos(x/H);

        x0 = 0; y0 = 0;

        xlapm = x0 + x;
        ylapm = y0 + H;

        xtrue = x0 + L*np.cos(th);
        ytrue = y0 + L*np.sin(th);

        plt.plot([x0, xlapm], [y0, ylapm], linewidth=2, label="CoM - LAPM");
        plt.plot([x0, xtrue], [y0, ytrue], linestyle=':', label="CoM - True");
        plt.plot([xlapm], [ylapm], marker='o',
         markerfacecolor='k', markeredgecolor='k', markersize=m/10);
        plt.plot([xtrue], [ytrue], marker='o',
         markerfacecolor='g', markeredgecolor='g', markersize=m/15);

        plt.plot([x0-L/4, x0+L/4], [y0, y0], 'k');
        plt.plot([xlapm, xlapm], [y0-d/2, y0+d/2], color='g', label="CP - Current");
        plt.plot([x0-d, x0-d], [y0-d/2, y0+d/2], 'k');
        plt.plot([x0+d, x0+d], [y0-d/2, y0+d/2], 'k');

        plt.title("TPM Simulation: t = {:.3f}".format(T[i]));
        plt.ylim(yLimits);  plt.xlim(xLimits);
        plt.legend();
        plt.grid();
        plt.pause(dt);

    input("Press enter to close animation...");

    return 1;

def plotStates_lapm(T, q):
    qT = np.transpose(q);

    fig, statePlot = plt.subplots(1,2);

    # CoM x-distance
    statePlot[0].plot(T, qT[0], label="x");
    statePlot[0].plot(T, qT[2], label="dx");
    statePlot[0].set_title("Position (x)");
    statePlot[0].legend();
    statePlot[0].grid();

    # angular momentum
    statePlot[1].plot(T, qT[1], label="Lc");
    statePlot[1].plot(T, qT[3], label="dLc");
    statePlot[1].set_title("Angular Momentum (L)");
    statePlot[1].legend();
    statePlot[1].grid();

    return statePlot;

def plotInputs_lapm(T, u, id=-1):
    if id == -1:
        Tspan = T;
        uT = np.transpose(u);
    else:
        P = int(len(u[0])/2);
        dt = T[1] - T[0];

        Tspan = [i*dt for i in range(P)];

        ua = [u[id+1][i] for i in range(0,2*(P),2)];
        Lc = [u[id+1][i] for i in range(1,2*(P),2)];

        uT = np.transpose([[0, 0] for i in range(P)]);
        uT[0] = ua;  uT[1] = Lc;

    fig, inputPlot = plt.subplots(1,2);

    inputPlot[0].plot(Tspan, uT[0], label="Actual ua");
    inputPlot[0].set_title("Ankle Torque Trend")
    inputPlot[0].set_ylabel("Torque [Nm]");
    inputPlot[0].set_xlabel("Time [s]");
    inputPlot[0].grid();

    inputPlot[1].plot(Tspan, uT[1], label="Actual Lc");
    inputPlot[1].set_title("Angular Momentum Trend")
    inputPlot[1].set_ylabel("Angular Momentum [Nm/s]");
    inputPlot[1].set_xlabel("Time [s]");
    inputPlot[1].grid();

    return inputPlot;

def plotMPCComparison_lapm(T, u, id=0):
    P = int(len(u[0])/2);
    dt = T[1] - T[0];
    T_mpc = [i*dt for i in range(P)];

    ua_mpc = [u[id+1][i] for i in range(0,2*(P),2)];
    Lc_mpc = [u[id+1][i] for i in range(1,2*(P),2)];

    mpcPlot = plotInputs_lapm(T, u, id);

    mpcPlot[0].plot(T_mpc, ua_mpc, linestyle='dashdot', label="MPC ua (t0={:.3f})".format(dt*(id)));
    mpcPlot[1].plot(T_mpc, Lc_mpc, linestyle='dashdot', label="MPC Lc (t0={:.3f})".format(dt*(id)));
    mpcPlot[0].legend();  mpcPlot[1].legend();

    return mpcPlot;

def plotCost_lapm(T, C):
    fig, costPlot = plt.subplots();

    costPlot.plot(T, C);
    costPlot.set_title("MPC Cost Trend");
    costPlot.set_ylabel("Cost");
    costPlot.set_xlabel("Time [s]");
    costPlot.grid();

    return costPlot;

def plotBrkFreq_lapm(brk):
    fig, brkFreqPlot = plt.subplots();

    unique, counts = np.unique(brk[1:], return_counts=1);

    for i in range(len(unique)):
        brkFreqPlot.bar([unique[i], unique[i]], [0, counts[i]], linewidth=3);

    plt.xlim([np.min(brk)-0.5, np.max(brk)+0.5]);
    plt.xticks(unique);

    return brkFreqPlot;

def plotRunTime_lapm(T, t):
    fig, runTimePlot = plt.subplots();

    aveRunTime = np.mean(t);

    runTimePlot.plot(T, t, label="Calc. Trend");
    runTimePlot.plot([T[0], T[-1]], [aveRunTime, aveRunTime], label="Average Calc. Time");
    runTimePlot.set_title("MPC Computation Runtime Trend");
    runTimePlot.set_ylabel("Calc. Time [ms]");
    runTimePlot.set_xlabel("Time [s]");
    runTimePlot.legend();
    runTimePlot.grid();

    return runTimePlot;

def saveResults_lapm(filename, inputs, mpc_var, mpc_results):
    with open(filename, "wb") as save_file:
        pickle.dump([inputs, mpc_var, mpc_results], save_file);
        # pickle.dump(mpc_var, save_file);
        # pickle.dump(mpc_results, save_file);

    return 1;

def loadResults_lapm(filename):
    with open(filename, "rb") as load_file:
        inputs, mpc_var, mpc_results = pickle.load(load_file);
        # mpc_var     = pickle.load(load_file);
        # mpc_results = pickle.load(load_file);

    return (inputs, mpc_var, mpc_results);

def staticPlots_lapm(mpc_results):
    T = mpc_results[0];
    q = mpc_results[1];
    u = mpc_results[2];
    C = mpc_results[3];
    n = mpc_results[4];
    brk = mpc_results[5];
    t = mpc_results[6];

    statePlot = plotStates_lapm(T, q);
    inputPlot = plotInputs_lapm(T, u);
    costPlot  = plotCost_lapm(T, C);
    brkFreqPlot = plotBrkFreq_lapm(brk);
    runTimePlot = plotRunTime_lapm(T, t);
    plt.show(block=0);

    input("Press enter to close static plots...");
    plt.close('all');

    return 1;

def reportResults_lapm(inputs, mpc_var, mpc_results):
    T = mpc_results[0];
    q = mpc_results[1];

    ans = input("\nSee state, input, and cost plots? [y/n] ");
    if ans == 'y':
        staticPlots_lapm(mpc_results);

    ans = input("\nSee animation? [y/n] ");
    if ans == 'y':
        animation_lapm(T, q, inputs);

    return 1;
