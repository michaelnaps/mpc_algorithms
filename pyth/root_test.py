import math
import random as rm
import numpy as np
import matplotlib.pyplot as plt
import nno
from modeuler import *
from statespace_lapm import *

def Cq(qd, q):
   Cq = [
      100*(qd[0] - q[0]) + (qd[2] - q[2]),
      100*(qd[1] - q[1]) + (qd[3] - q[3])
   ];
   return Cq;

def Cu(u, du):
   Cu = [
      1e-7*(du[0])**2 + (u[0]/1000)**4,
      1e-7*(du[1])**2 + (u[1]/1000)**4,
   ];
   return Cu;

class mpc_var:
   sim_time     = 1;
   model        = statespace_lapm;
   state_cost   = Cq;
   input_cost   = Cu;
   PH_length    = 1;
   knot_length  = 4;
   time_step    = 0.025;
   appx_zero    = 1e-6;
   step_size    = 1e-3;
   num_joints   = 1;
   num_ssvar    = 2;
   num_inputs   = 2;
   input_bounds = [5000, 5000];
   des_config   = [0, 0, 0, 0];

class inputs:
   # Constants and State Variables
   num_joints           = 3;
   gravity_acc          = -9.81;
   damping_coefficients = [0];
   joint_masses         = [10];
   link_lengths         = [1.0];

q0 = [0-0.01, 0, 0, 0];
u0 = [0, 0];

mpc_results = nno.mpc_root(mpc_var, q0, u0, inputs);

T = mpc_results[0];
q = mpc_results[1];
u = mpc_results[2];
C = mpc_results[3];

input("Press Enter for animation...");

animation_lapm(T, q, inputs);

nno.save_results("prevRun_data.pickle", mpc_results);
