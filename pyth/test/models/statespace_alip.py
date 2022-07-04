import numpy as np
import matplotlib.pyplot as plt
import pickle

def statespace_alip(q, u, inputs):
    # constant variables
    m = inputs.joint_masses[0];
    H = inputs.link_lengths[0];
    g = inputs.gravity_acc;

    # state and input variables
    x  = q[0];  L  = q[1];
    ua = 0;     Lc = u[0];

    # solve for statespace
    ddq = [
        L/(m*H) + Lc/(m*H),
        ua + m*g*x
    ];

    return [ddq[0], ddq[1]];

def animation_alip(T, q, inputs):
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

        xalip = x0 + x;
        yalip = y0 + H;

        xtrue = x0 + L*np.cos(th);
        ytrue = y0 + L*np.sin(th);

        plt.plot([x0, xalip], [y0, yalip], linewidth=2, label="CoM - ALIP");
        plt.plot([x0, xtrue], [y0, ytrue], linestyle=':', label="CoM - True");
        plt.plot([xalip], [yalip], marker='o',
         markerfacecolor='k', markeredgecolor='k', markersize=m/10);
        plt.plot([xtrue], [ytrue], marker='o',
         markerfacecolor='g', markeredgecolor='g', markersize=m/15);

        plt.plot([x0-L/4, x0+L/4], [y0, y0], 'k');
        plt.plot([xalip, xalip], [y0-d/2, y0+d/2], color='g', label="CP - Current");
        plt.plot([x0-d, x0-d], [y0-d/2, y0+d/2], 'k');
        plt.plot([x0+d, x0+d], [y0-d/2, y0+d/2], 'k');

        plt.title("TPM Simulation: t = {:.3f}".format(T[i]));
        plt.ylim(yLimits);  plt.xlim(xLimits);
        plt.legend();
        plt.grid();
        plt.pause(dt);

    input("Press enter to close animation...");

    return 1;

def plotStates_alip(T, q):
    qT = np.transpose(q);

    fig, statePlot = plt.subplots(1,2);

    # CoM x-distance
    statePlot[0].plot(T, qT[0], label="x");
    statePlot[0].set_title("Position (x)");
    statePlot[0].grid();

    # angular momentum
    statePlot[1].plot(T, qT[1], label="Lc");
    statePlot[1].set_title("Angular Momentum about the Pivot (L)");
    statePlot[1].grid();

    return statePlot;

def plotInputs_alip(T, u, id=-1):
    uT = np.transpose(u);

    if id == -1:
        Tspan = T;
    else:
        P = int(len(u[0])/2);
        dt = T[1] - T[0];
        Tspan = [i*dt for i in range(P)];

        uT = [uT[0][id:P+id], uT[1][id:P+id]];

    fig, inputPlot = plt.subplots();
    # fig, inputPlot = plt.subplots(1,2);

    inputPlot.plot(Tspan, uT[0], label="Actual Lc");
    inputPlot.set_title("Angular Momentum about the COM (L_c)")
    inputPlot.set_ylabel("Angular Momentum [Nm/s]");
    inputPlot.set_xlabel("Time [s]");
    inputPlot.grid();

    """
    inputPlot[0].plot(Tspan, uT[0], label="Actual ua");
    inputPlot[0].set_title("Ankle Torque (u_a)")
    inputPlot[0].set_ylabel("Torque [Nm]");
    inputPlot[0].set_xlabel("Time [s]");
    inputPlot[0].grid();

    inputPlot[1].plot(Tspan, uT[1], label="Actual Lc");
    inputPlot[1].set_title("Angular Momentum about the COM (L_c)")
    inputPlot[1].set_ylabel("Angular Momentum [Nm/s]");
    inputPlot[1].set_xlabel("Time [s]");
    inputPlot[1].grid();
    """

    return inputPlot;

def plotMPCComparison_alip(T, u, id=0):
    P = int(len(u[0])/2);
    dt = T[1] - T[0];
    T_mpc = [i*dt for i in range(P)];

    ua_mpc = [u[id+1][i] for i in range(0,2*(P),2)];
    Lc_mpc = [u[id+1][i] for i in range(1,2*(P),2)];

    mpcPlot = plotInputs_alip(T, u, id);

    mpcPlot[0].plot(T_mpc, ua_mpc, linestyle='dashdot', label="MPC ua (t0={:.3f})".format(dt*(id)));
    mpcPlot[1].plot(T_mpc, Lc_mpc, linestyle='dashdot', label="MPC Lc (t0={:.3f})".format(dt*(id)));
    mpcPlot[0].legend();  mpcPlot[1].legend();

    return mpcPlot;

def plotCost_alip(T, C):
    fig, costPlot = plt.subplots();

    costPlot.plot(T, C);
    costPlot.set_title("MPC Cost Trend");
    costPlot.set_ylabel("Cost");
    costPlot.set_xlabel("Time [s]");
    costPlot.grid();

    return costPlot;

def plotBrkFreq_alip(brk, explode_id=-1):
    # sort out -2 values
    Nbrk = len(brk);
    brklist = [];
    for i in range(Nbrk):
        if (brk[i] != -2):
            brklist.append(brk[i]);
        elif (brk[i-1] != -2):
            brklist.append(brk[i]);

    all_labels = [
        [-2, 'Diverged', 'firebrick'],
        [-1, 'Iter Break', 'darkorange'],
        [0, 'Zero Cost', '#1f77b4'],
        [1, 'FOOC', 'yellowgreen'],
        [2, 'Zero Change', 'mediumorchid']
    ];

    unique, counts = np.unique(brklist, return_counts=1);
    Nu = len(unique) - 1;

    labels = [];
    colors = [];
    explode = ();
    percent = [];
    for i in range(Nu):
        for j in range(len(all_labels)):
            if (unique[i] == all_labels[j][0]):
                labels.append(all_labels[j][1]);
                colors.append(all_labels[j][2]);
                percent.append(counts[i]/np.sum(counts));

                if (i==explode_id):
                    temp = (0.1,)
                    explode = explode + temp;
                else:
                    temp = (0.0,)
                    explode = explode + temp;

                break;

    fig, brkFreqPlot = plt.subplots(constrained_layout=True);
    # brkFreqPlot.pie(counts[:Nu], labels=labels, colors=colors, explode=explode, shadow=True, autopct='%.3f%%');

    brkFreqPlot.bar(unique[:Nu], percent, color=colors, zorder=5);
    brkFreqPlot.set_xticks(unique[:Nu]);
    brkFreqPlot.set_xticklabels(labels);
    brkFreqPlot.grid(zorder=0);

    return brkFreqPlot;

def plotRunTime_alip(T, t, title=1):
    fig, runTimePlot = plt.subplots(constrained_layout=True);

    aveRunTime = np.mean(t);

    runTimePlot.plot(T, t, label="Calc. Trend");
    runTimePlot.plot([T[0], T[-1]], [aveRunTime, aveRunTime], label="Average Calc. Time");
    runTimePlot.set_ylabel("Calc. Time [ms]");
    runTimePlot.set_xlabel("Time [s]");
    runTimePlot.legend();
    runTimePlot.grid();
    if title:  runTimePlot.set_title("MPC Computation Runtime Trend");

    return runTimePlot;

def saveResults_alip(filename, sim_results):
    with open(filename, "wb") as save_file:
        pickle.dump(sim_results, save_file);

    print("\nResults saved...");
    return 1;

def loadResults_alip(filename):
    with open(filename, "rb") as load_file:
        sim_results = pickle.load(load_file);

    return sim_results;

def staticPlots_alip(sim_results):
    T = sim_results[0];
    q = sim_results[1];
    u = sim_results[2];
    C = sim_results[3];
    n = sim_results[4];
    brk = sim_results[5];
    t = sim_results[6];

    statePlot = plotStates_alip(T, q);
    inputPlot = plotInputs_alip(T, u);
    costPlot  = plotCost_alip(T, C);
    brkFreqPlot = plotBrkFreq_alip(brk);
    runTimePlot = plotRunTime_alip(T, t);
    plt.show(block=0);

    input("Press enter to close static plots...");
    plt.close('all');

    return 1;

def reportResults_alip(sim_results, inputs=0):
    T = sim_results[0];
    q = sim_results[1];

    ans = input("\nSee state, input, and cost plots? [y/n] ");
    if (ans == 'y'):  staticPlots_alip(sim_results);

    if (inputs != 0):
        ans = input("\nSee animation? [y/n] ");
        if (ans == 'y'):  animation_alip(T, q, inputs);

    return 1;
