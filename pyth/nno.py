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

      if output:  print("Elapsed Time:  %0.3f\n" % (t_run[i]));

      # inverse dynamics: lapm -> 3link (digit)

      qlist[i] = modeuler(mpc_var, qlist[i-1], ulist[i][:N], inputs)[1][1];

   return (T, qlist, ulist, Clist, nlist, brklist, t_run);

def newtons(mpc_var, q0, u0, uinit, inputs, output=0):
   # MPC constants initialization
   P    = mpc_var.PH_length;
   N    = mpc_var.num_inputs;
   eps  = mpc_var.appx_zero;
   umax = mpc_var.input_bounds;

   # loop variable setup
   uc = uinit;
   Cc = cost(mpc_var, q0, u0, uc, inputs);
   un = uc;  Cn = Cc;

   if output:
      print("Opt. Start:");
      print("Initial Cost: ", Cc);

   count = 1;
   brk = 0;
   while (Cc > eps):
   	  # calculate the gradient around the current input
      g = gradient(mpc_var, q0, u0, uc, inputs);
      gnorm = np.sqrt(np.sum([g[i]**2 for i in range(N)]));

      # check if gradient-norm is an approx. of zero
      if (gnorm < eps):
         brk = 1;
         break;

      # calculation the hessian around the current input
      H = hessian(mpc_var, q0, u0, uc, inputs);

      # calculate the next iteration of the input
      udn = np.linalg.lstsq(H, g, rcond=None)[0];
      un = [uc[i] - udn[i] for i in range(P*N)];

      # simulate and calculate the new cost value
      Cn = cost(mpc_var, q0, u0, un, inputs);
      count += 1;  # iterate the loop counter

      if output:
         # print("Gradient:  ", g);
         print("|g|:          ", gnorm);
         print("New Cost:     ", Cn);
         print("New Input: ");
         for i in range(P*N):  print("              ", un[i]);

      # break conditions
      if (count == 6):
         brk = -1;
         break;

      # update loop variables
      uc = un;  Cc = Cn;

   for i in range(P*N):
      if un[i] > umax[i]:
         un[i] = umax[i];
      elif un[i] < -umax[i]:
         un[i] = -umax[i];

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
      C[i] = C[i] + Cq(qd, q[i]);
      if i != P:
         C[i] = C[i] + Cu(uc[i], du[i]);
         C[i] = C[i] + Ccmp(uc[i], inputs);

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


def hessian(mpc_var, q, u0, u, inputs):
   # variable setup
   N = mpc_var.num_inputs*mpc_var.PH_length;
   h = mpc_var.step_size;
   H = [[0 for i in range(N)] for j in range(N)];

   for i in range(N):
      un1 = [u[j] - (i==j)*h for j in range(N)];
      up1 = [u[j] + (i==j)*h for j in range(N)];

      gn1 = gradient(mpc_var, q, u0, un1, inputs, i);
      gp1 = gradient(mpc_var, q, u0, up1, inputs, i);

      # enforce symmetry
      for j in range(i, N):
          H[i][j] = (gp1[j] - gn1[j])/(2*h);
          H[j][i] = H[i][j];

   return H;

def save_results(filename, mpc_results):
   with open(filename, 'wb') as save_file:
      pickle.dump(mpc_results, save_file);

   return 1;

def load_results(filename):
   with open(filename, 'rb') as save_file:
      mpc_results = pickle.load(save_file);

   return mpc_results;

def report_results(filename):
    mpc_results = load_results(filename);

    T = mpc_results[0];
    q = mpc_results[1];
    u = mpc_results[2];
    C = mpc_results[3];
    brk = mpc_results[5];
    t = mpc_results[6];

    ans = input("\nSee state, input, and cost plots? [y/n] ");
    if ans == 'y':
        statePlot = plotStates_lapm(T, q);
        inputPlot = plotInputs_lapm(T, u);
        costPlot  = plotCost_lapm(T, C);
        brkFreqPlot = plotBrkFreq_lapm(brk);
        runTimePlot = plotRunTime_lapm(T, t);
        plt.show();

    ans = input("\nSee animation? [y/n] ");
    if ans == 'y':
       animation_lapm(T, q, inputs);

   return mpc;
