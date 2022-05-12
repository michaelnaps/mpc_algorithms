# Package: nno.py
# Created by: Michael Napoli
# Created on: 3/27/2022
# Last modified on: 3/27/2022


import numpy as np
import math
import time
import pickle
from modeuler import *


def mpc_root(mpc_var, q0, u0, inputs, output=0):
    # MPC variable declaration
    N  = mpc_var.num_inputs;
    P  = mpc_var.PH_length;
    dt = mpc_var.time_step;
    eps = mpc_var.appx_zero;
    tspan = mpc_var.sim_time;

    # simulation time variables
    Nt = int(tspan/dt+1);
    T = [i*dt for i in range(Nt)];

    # state matrices declarations
    qlist = [0 for i in range(Nt)];
    qlist[0] = q0;

    # return variables
    ulist = [0 for i in range(Nt)];
    Clist = [0 for i in range(Nt)];
    nlist = [0 for i in range(Nt)];
    brklist = [0 for i in range(Nt)];
    t_run = [0 for i in range(Nt)];

    ulist[0] = u0;

    for i in range(1,Nt):
        if output:  print("\nTime: %0.3f" % (T[i]));

        t = time.time();
        opt_results = newtons(mpc_var, qlist[i-1], ulist[i-1][:N], ulist[i-1], inputs, output);
        elapsed = time.time() - t;

        ulist[i]   = opt_results[0];
        Clist[i]   = opt_results[1];
        nlist[i]   = opt_results[2];
        brklist[i] = opt_results[3];
        t_run[i]   = 1000*elapsed;

        if output:  print("Elapsed Time:\n              ", t_run[i]);

        # inverse dynamics: lapm -> 3link (digit)

        qlist[i] = modeuler(mpc_var, qlist[i-1], ulist[i][:N], inputs)[1][1];

    return (T, qlist, ulist, Clist, nlist, brklist, t_run);

def newtons(mpc_var, q0, u0, uinit, inputs, output=0):
    # MPC constants initialization
    P    = mpc_var.PH_length;
    N    = mpc_var.num_inputs;
    eps  = mpc_var.appx_zero;
    imax = mpc_var.max_iter;

    # step size coefficient choice
    alpha    = mpc_var.alpha;
    a_method = mpc_var.a_method;

    # loop variable setup
    uc = uinit;
    Cc = cost(mpc_var, q0, u0, uc, inputs);
    un = uc;  Cn = Cc;

    if output:
        print("Opt. Start:");
        print("Initial Cost: ", Cc);

    count = 1;
    brk = -2*np.isnan(Cc);
    while (Cc > eps):
        # calculate the gradient around the current input
        g = gradient(mpc_var, q0, u0, uc, inputs);
        gnorm = np.sqrt(np.sum([g[i]**2 for i in range(N)]));

        # check if gradient-norm is an approx. of zero
        if (gnorm < eps):
            brk = 1;
            break;

        # calculate the next iteration of the input
        if (a_method == "bkl"):  un = alpha_bkl(g, Cc, mpc_var, q0, u0, uc, inputs)[0];
        elif (a_method == "bis"):  un = alpha_bis(g, Cc, mpc_var, q0, u0, uc, inputs)[0];
        else:  uave = [uc[i] - alpha*g[i] for i in range(P*N)];

        # simulate and calculate the new cost value
        Cn = cost(mpc_var, q0, u0, un, inputs);
        count += 1;  # iterate the loop counter

        if (np.isnan(Cn)):
            brk = -2;
            break;

        if output:
            # print("Gradient:  ", g);
            print("|g|:          ", gnorm);
            print("New Cost:     ", Cn);
            print("New Input: ");
            for i in range(0, N*P, 2):
                print("              ", un[i], un[i+1]);

        # break conditions
        if (count > imax):
            brk = -1;
            break;

        # update loop variables
        uc = un;  Cc = Cn;

    return (un, Cn, count, brk);


def cost(mpc_var, q0, u0, u, inputs):
    # MPC constants
    N  = mpc_var.num_ssvar;
    Nu = mpc_var.num_inputs;
    P  = mpc_var.PH_length;
    Cq = mpc_var.cost_state;
    Cu = mpc_var.cost_input;
    Ccmp = mpc_var.cost_CMP;
    qd = mpc_var.des_config;

    # reshape input variable
    uc = np.reshape(u, [P, Nu]);

    # calculate change in input
    du = [[0 for j in range(Nu)] for i in range(P)];
    du[0] = [uc[0][i] - u0[i] for i in range(Nu)];
    for i in range(1, P):
        for j in range(Nu):
            du[i][j] = uc[i][j] - uc[i-1][j];

    # Cost of each input over the designated windows
    # simulate over the prediction horizon and sum cost
    q = [[0 for i in range(2*N)] for j in range(P+1)];
    q[0] = q0;
    for i in range(P):
        q[i+1] = modeuler(mpc_var, q[i], uc[i], inputs)[1][-1];

    C = [0 for i in range(P+1)];

    for i in range(P+1):
        C[i] = C[i] + Cq(qd, q[i]);  # state cost
        if i != P:
            C[i] = C[i] + Cu(uc[i], du[i], inputs);  # input costs
            C[i] = C[i] + Ccmp(uc[i], inputs);  # CMP costs

    return np.sum(C);


def gradient(mpc_var, q, u0, u, inputs, rownum=1):
    # variable setup
    N = mpc_var.num_inputs*mpc_var.PH_length;
    h = mpc_var.step_size;
    g = [0 for i in range(N)];

    for i in range(rownum-1, N):
        un1 = [u[j] - (i==j)*h for j in range(N)];
        up1 = [u[j] + (i==j)*h for j in range(N)];

        Cn1 = cost(mpc_var, q, u0, un1, inputs);
        Cp1 = cost(mpc_var, q, u0, up1, inputs);

        g[i] = (Cp1 - Cn1)/(2*h);

    return g;

def alpha_bkl(g, C0, mpc_var, q0, u0, uc, inputs):
    P = mpc_var.PH_length;
    N = mpc_var.num_inputs;
    w = mpc_var.bkl_shrink;
    a = mpc_var.alpha;
    abkl = a;

    count = 0;
    brk = -1;
    while (count != 1000):
        ubkl = [uc[i] - abkl*g[i] for i in range(P*N)];
        Cbkl = cost(mpc_var, q0, u0, ubkl, inputs);
        count += 1;

        if (Cbkl < C0):
            brk = 0;
            break;

        abkl *= w;

    print("Alpha: ", abkl);
    return (ubkl, Cbkl, abkl, count, brk);

def alpha_bis(g, C0, mpc_var, q0, u0, uc, inputs):
    P = mpc_var.PH_length;
    N = mpc_var.num_inputs;
    a = mpc_var.alpha;
    eps = mpc_var.appx_zero;

    # bisection variable setup
    alow = a[0];  ahgh = a[1];
    aave = (ahgh + alow)/2;

    ulow = [uc[i] - alow*g[i] for i in range(P*N)];
    uhgh = [uc[i] - ahgh*g[i] for i in range(P*N)];
    uave = [uc[i] - aave*g[i] for i in range(P*N)];

    Clow = cost(mpc_var, q0, u0, ulow, inputs);
    Chgh = cost(mpc_var, q0, u0, uhgh, inputs);
    Cave = cost(mpc_var, q0, u0, uave, inputs);

    # bisection loop
    count = 0;
    brk = 0;
    while (Cave > eps):
        if (Clow < Chgh):
            ahgh = aave;
            Chgh = Cave;
        else:
            alow = aave;
            Clow = Cave;

        aave = (ahgh + alow)/2;
        uave = [uc[i] - aave*g[i] for i in range(P*N)];
        Cave = cost(mpc_var, q0, u0, uave, inputs);

        if ((ahgh - alow) < eps):
            brk = 1;
            break;

        count += 1;

        if (count == 1000):
            brk = -1;
            break;

    print("Alpha: ", aave);
    return (uave, Cave, aave, count, brk);
