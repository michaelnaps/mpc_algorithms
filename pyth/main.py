import math
import numpy as np
import nno
from modeuler import *
import statespace_3link as model

def CF(qd, q):
   Cqu = [
      100*((np.cos(qd(1)) - np.cos(q(1)))^2 + (np.sin(qd(1)) - np.sin(q(1)))^2) + (qd(2) - q(2))^2, # + 1e-7*(du(1))^2;  % cost of Link 1
      100*((np.cos(qd(2)) - np.cos(q(3)))^2 + (np.sin(qd(2)) - np.sin(q(3)))^2) + (qd(4) - q(4))^2, # + 1e-7*(du(2))^2;  % cost of Link 2
      100*((np.cos(qd(3)) - np.cos(q(5)))^2 + (np.sin(qd(3)) - np.sin(q(5)))^2) + (qd(6) - q(6))^2, # + 1e-7*(du(3))^2;  % cost of Link 3
   ];
   return Cqu;

class mpc_var:
   PH_length  = 40;
   time_step  = 0.025;
   appx_zero  = 1e-6;
   cost_func  = CF;
   des_angles = [math.pi/4, -math.pi/2, math.pi/2];
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

q = nno.mpc_root(mpc_var, statespace_3link, q0, inputs);

#qc = modeuler(mpc_var, q0, u, statespace_3link);
