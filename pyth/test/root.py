import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');

import nno
from statespace_lapm import *

def Cq(qd, q):
   Cq = [
      100*(qd[0] - q[0])**2 + (qd[2] - q[2])**2,
       10*(qd[1] - q[1])**2 + (qd[3] - q[3])**2
   ];
   
   return np.sum(Cq);

def Cu(u, du):
   umax = [60, 150];
   
   Cu = [
      1e-5*(du[0])**2 - np.log(umax[0]**2 - u[0]**2) + np.log(umax[0]**2),
      1e-5*(du[1])**2# - np.log(umax[1]**2 - u[1]**2) + np.log(umax[1]**2)
   ];
   
   return np.sum(Cu);
   
def Ccp(u, inputs):
   dmax = inputs.CP_distance;
   g = inputs.gravity_acc;
   m = inputs.joint_masses[0];
   
   Ccp = [
      -np.log((m*g*dmax)**2 - u[0]**2) + np.log((m*g*dmax)**2),
      0
   ];
   
   return np.sum(Ccp);

class mpc_var:
   sim_time     = 5;
   model        = statespace_lapm;
   state_cost   = Cq;
   input_cost   = Cu;
   CP_cost      = Ccp;
   PH_length    = 2;
   knot_length  = 4;
   time_step    = 0.025;
   appx_zero    = 1e-6;
   step_size    = 1e-3;
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
   CP_distance          = 0.1;

q0 = [0-0.075, 0, 0, 0];
u0 = [0 for i in range(mpc_var.num_inputs*mpc_var.PH_length)];

mpc_results = nno.mpc_root(mpc_var, q0, u0, inputs, 1);

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
