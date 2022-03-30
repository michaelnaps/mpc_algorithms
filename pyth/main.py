import math
import numpy as np
import nno
from modeuler import *
from statespace_3link import *

def CF(qd, q):
   Cqu = [
      100*((np.cos(qd[0]) - np.cos(q[0]))**2 + (np.sin(qd[0]) - np.sin(q[0]))**2) + (qd[1] - q[1])**2, # + 1e-7*(du[0])**2;  % cost of Link 1
      100*((np.cos(qd[2]) - np.cos(q[2]))**2 + (np.sin(qd[2]) - np.sin(q[2]))**2) + (qd[3] - q[3])**2, # + 1e-7*(du[1])**2;  % cost of Link 2
      100*((np.cos(qd[4]) - np.cos(q[4]))**2 + (np.sin(qd[4]) - np.sin(q[4]))**2) + (qd[5] - q[5])**2, # + 1e-7*(du[2])**2;  % cost of Link 3
   ];
   return Cqu;

class mpc_var:
   PH_length  = 40;
   time_step  = 0.025;
   appx_zero  = 1e-6;
   step_size  = 1e-3;
   cost_func  = CF;
   des_angles = [math.pi/2, 0, math.pi, 0, math.pi, 0];
   model      = statespace_3link;

class inputs:
   # Constants and State Variables
   gravity_acc          = 9.81;
   max_torque           = [3000, 3000, 3000];
   damping_coefficients = [500, 500, 500];
   joint_masses         = [15, 15, 60];
   link_lengths         = [0.5, 0.5, 1.0];

q0 = [math.pi/2, 0, math.pi, 0, math.pi, 0];
u0 = [0, 0, 0];

for i in range(100):
   H = nno.hessian(mpc_var, statespace_3link, q0, [i,i,i], inputs);
   print(H);
