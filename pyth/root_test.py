import math
import numpy as np
import matplotlib.pyplot as plt
import nno
from modeuler import *
from statespace_3link import *

def CF(qd, q, du):
   Cqu = [
      100*((np.cos(qd[0]) - np.cos(q[0]))**2 + (np.sin(qd[0]) - np.sin(q[0]))**2) + (qd[1] - q[1])**2 + 1e-7*(du[0])**2,  # cost of Link 1
      100*((np.cos(qd[2]) - np.cos(q[2]))**2 + (np.sin(qd[2]) - np.sin(q[2]))**2) + (qd[3] - q[3])**2 + 1e-7*(du[1])**2,  # cost of Link 2
      100*((np.cos(qd[4]) - np.cos(q[4]))**2 + (np.sin(qd[4]) - np.sin(q[4]))**2) + (qd[5] - q[5])**2 + 1e-7*(du[2])**2,  # cost of Link 3
   ];
   return Cqu;

class mpc_var:
   sim_time     = 10;
   model        = statespace_3link;
   cost_func    = CF;
   PH_length    = 400;
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

q0 = [math.pi/2-0.01, 0, 0, 0, 0, 0];
u0 = [0, 0, 0];

T, q = modeuler(mpc_var, q0, u0, inputs);

animPlot = animation_3link(T, q);

"""
mpc_results = nno.mpc_root(mpc_var, q0, u0, inputs);

T = mpc_results[0];
q = mpc_results[1];
u = mpc_results[2];
C = mpc_results[3];

statePlot = plotStates_3link(T, q);
inputPlot = plotInputs_3link(T, u);
costPlot  = plotCost_3link(T, C);

plt.show();

"""

#statePlot = plotStates_3link(T, q);
plt.show();
