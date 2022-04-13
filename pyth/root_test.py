import math
import random as rm
import numpy as np
import matplotlib.pyplot as plt
import nno
from modeuler import *
from statespace_3link import *

def Cq(qd, q):
   Cq = [
      100*(qd[0] - q[0])**2 + (qd[3] - q[3])**2,# + 1e-7*(du[0])**2,  # cost of Link 1
      100*(qd[1] - q[1])**2 + (qd[4] - q[4])**2,# + 1e-7*(du[1])**2,  # cost of Link 2
      100*(qd[2] - q[2])**2 + (qd[5] - q[5])**2,# + 1e-7*(du[2])**2,  # cost of Link 3
   ];
   return Cq;

def Cu(u, du):
   Cu = [
      1e-7*(du[0])**2 + (u[0]/300)**4,
      1e-7*(du[1])**2 + (u[1]/300)**4,
      # 1e-7*(du[2])**2 + (u[2]/1500)**4,
   ];
   return Cu;

class mpc_var:
   sim_time     = 1;
   model        = statespace_3link;
   state_cost   = Cq;
   input_cost   = Cu;
   PH_length    = 4;
   knot_length  = 4;
   time_step    = 0.025;
   appx_zero    = 1e-6;
   step_size    = 1e-3;
   num_joints   = 3;
   num_inputs   = 2;
   input_bounds = [10000 for i in range(num_inputs*PH_length)];
   des_config   = [math.pi/2, 0, 0, 0, 0, 0];

class inputs:
   # Constants and State Variables
   num_joints           = 3;
   gravity_acc          = 9.81;
   damping_coefficients = [10, 10, 10];
   joint_masses         = [1, 1, 1];
   link_lengths         = [0.5, 0.5, 1.0];

q0 = [-math.pi/2, 0, 0, 0, 0, 0];
u0 = [rm.randint(-10,10) for i in range(mpc_var.num_inputs*mpc_var.PH_length)];

mpc_results = nno.mpc_root(mpc_var, q0, u0, inputs);

T = mpc_results[0];
q = mpc_results[1];
u = mpc_results[2];
C = mpc_results[3];

print("\nSimulation complete!");
input("Press Enter for performance plots...");

statePlot = plotStates_3link(T, q);
inputPlot = plotInputs_3link(T, u);
costPlot  = plotCost_3link(T, C);
plt.show();

input("Press Enter for animation...");

animation_3link(T, q, inputs);

nno.save_results("prevRun_data.pickle", mpc_results);
