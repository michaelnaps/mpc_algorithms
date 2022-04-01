import math
import numpy as np
import matplotlib.pyplot as plt
import nno
from modeuler import *
from statespace_3link import *

def CF(qd, q, du):
   Cqu = [
      100*((np.cos(qd[0]) - np.cos(q[0]))**2 + (np.sin(qd[0]) - np.sin(q[0]))**2) + (qd[1] - q[1])**2,# + 1e-7*(du[0])**2,  # cost of Link 1
      100*((np.cos(qd[2]) - np.cos(q[2]))**2 + (np.sin(qd[2]) - np.sin(q[2]))**2) + (qd[3] - q[3])**2,# + 1e-7*(du[1])**2,  # cost of Link 2
      100*((np.cos(qd[4]) - np.cos(q[4]))**2 + (np.sin(qd[4]) - np.sin(q[4]))**2) + (qd[5] - q[5])**2,# + 1e-7*(du[2])**2,  # cost of Link 3
   ];
   return Cqu;

class mpc_var:
   sim_time     = 10;
   model        = statespace_3link;
   cost_func    = CF;
   PH_length    = 4;
   time_step    = 0.025;
   appx_zero    = 1e-6;
   step_size    = 1e-3;
   input_bounds = [3000, 3000, 3000];
   des_config   = [math.pi/2, 0, 0, 0, 0, 0];

class inputs:
   # Constants and State Variables
   gravity_acc          = 9.81;
   damping_coefficients = [500, 500, 500];
   joint_masses         = [15, 15, 60];
   link_lengths         = [0.5, 0.5, 1.0];

q0 = [math.pi/4, 0, math.pi/2, 0, -math.pi/4, 0];
u0 = [0, 0, 0];

opt_results = nno.mpc_root(mpc_var, q0, u0, inputs);

print(opt_results);

T = opt_results[0];
q = opt_results[1];
u = opt_results[2];
C = opt_results[3];
n = opt_results[4];
brk = opt_results[5];

statePlot = plotStates_3link(T, q);
inputPlot = plotInputs_3link(T, u);

plt.show();


