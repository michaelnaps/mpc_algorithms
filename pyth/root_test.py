import math
import random as rm
import numpy as np
import matplotlib.pyplot as plt
import nno
from modeuler import *
from statespace_lapm import *

def Cq(qd, q):
   Cq = [
      100*(qd[0] - q[0])**2 + (qd[2] - q[2])**2,
      100*(qd[1] - q[1])**2 + (qd[3] - q[3])**2
   ];
   return Cq;

def Cu(u, du):
   Cu = [
      1e-5*(du[0])**2 - np.log(60**2 - u[0]**2) + np.log(60**2),
      1e-5*(du[1])**2, # + (u[1]/1000)**4,
   ];
   return Cu;

class mpc_var:
   sim_time     = 10;
   model        = statespace_lapm;
   state_cost   = Cq;
   input_cost   = Cu;
   PH_length    = 2;
   knot_length  = 4;
   time_step    = 0.025;
   appx_zero    = 1e-6;
   step_size    = 1e-3;
   num_joints   = 1;
   num_ssvar    = 2;
   num_inputs   = 2;
   input_bounds = [1000 for i in range(num_inputs*PH_length)];
   des_config   = [0, 0, 0, 0];

class inputs:
   # Constants and State Variables
   num_joints           = 3;
   gravity_acc          = -9.81;
   damping_coefficients = [0];
   joint_masses         = [80];
   link_lengths         = [2.0];

q0 = [0-0.05, 0, 0, 0];
u0 = [0 for i in range(mpc_var.num_inputs*mpc_var.PH_length)];

mpc_results = nno.mpc_root(mpc_var, q0, u0, inputs);

print("\nBreak List: ", mpc_results[5]);

T = mpc_results[0];
q = mpc_results[1];
u = mpc_results[2];
C = mpc_results[3];

statePlot = plotStates_lapm(T, q);
inputPlot = plotInputs_lapm(T, u);
costPlot  = plotCost_lapm(T, C);
plt.show();

ans = input("\nSee animation? [y/n] ");
if ans == 'y':
   animation_lapm(T, q, inputs);

nno.save_results("prevRun_data.pickle", mpc_results);
