# Package: nno.py
# Created by: Michael Napoli
# Created on: 3/27/2022
# Last modified on: 3/27/2022


import numpy as np
import math
from modeuler import *


def mpc_root(mpc_var, q0, u0, inputs):
   # MPC variable declaration
   tspan = mpc_var.sim_time;
   P  = mpc_var.PH_length;
   dt = mpc_var.time_step;
   
   # simulation time variables
   Nt = int(tspan/dt+1);
   T = [i*dt for i in range(Nt)];
   
   # state matrices declarations
   N = int(len(mpc_var.des_config)/2)
   qlist = [[0 for i in range(N)] for j in range(Nt)];
   qlist[0] = q0;
   
   # return variables
   ulist = [[0 for i in range(N)] for j in range(Nt)];
   Clist = [0 for i in range(Nt)];
   nlist = [0 for i in range(Nt)];
   brklist = [0 for i in range(Nt)];
   
   ulist[0] = u0;

   for i in range(1,Nt):
      opt_results = newtons(mpc_var, qlist[i-1], ulist[i-1], ulist[i-1], inputs);
      ulist[i] = opt_results[0];
      Clist[i] = opt_results[1];
      nlist[i] = opt_results[2];
      brklist[i] = opt_results[3];
      
      print(qlist[i]);
      qlist[i] = modeuler(mpc_var, qlist[i-1], ulist[i], inputs)[1];
   
   return (T, qlist, ulist, Clist, nlist, brklist);
   
def newtons(mpc_var, q0, u0, uinit, inputs):
   # variable setup
   N  = int(len(q0)/2);
   eps = mpc_var.appx_zero;
   uc  = uinit;
   Cc  = cost(mpc_var, q0, u0, uc, inputs);
   un = uc;  Cn = Cc;
   
   count = 1;
   brk = 0;
   while(Cc != 0):
   	# calculate the gradient around the current input
      g = gradient(mpc_var, q0, u0, uc, inputs);
      gnorm = np.sqrt(np.sum([g[i]**2 for i in range(N)]))/N;
      
      # check if gradient is an approx. of zero
      if (gnorm < eps):
         brk = 1;
         break;
      
      # calculation the hessian around the current input
      H = hessian(mpc_var, q0, u0, uc, inputs);
      
      # calculate the next iteration of the input
      udn = np.linalg.lstsq(H, g, rcond=None)[0];
      un = [uc[i] - udn[i] for i in range(N)];
      
      # simulate and calculate the new cost value
      Cn = cost(mpc_var, q0, u0, un, inputs);
      count += 1;  # iterate the loop counter
      
      # break conditions
      if (count == 100):
         brk = -1;
         break;
      
      # update loop variables   
      uc = un;  Cc = Cn;
   
   return (un, Cn, count, brk);
   

def cost(mpc_var, q0, u0, u, inputs):
   # MPC constants
   P  = mpc_var.PH_length;
   Cq = mpc_var.cost_func;
   qd = mpc_var.des_config;
   
   # Cost of Constant Input
   # simulate over the prediction horizon and sum cost
   q  = modeuler(mpc_var, q0, u, inputs)[1];
   du = [u[i] - u0[i] for i in range(len(u))];

   C = [0 for i in range(len(u))];

   for i in range(P+1):
      C = C + Cq(qd, q[i], du);

   return np.sum(C);
   

def gradient(mpc_var, q, u0, u, inputs):
   # variable setup
   N = len(u);
   g = [0 for i in range(N)];
   h = mpc_var.step_size;
   
   for i in range(N):
      un1 = [u[j] - (i==j)*h for j in range(N)];
      up1 = [u[j] + (i==j)*h for j in range(N)];
      
      Cn1 = cost(mpc_var, q, u0, un1, inputs);
      Cp1 = cost(mpc_var, q, u0, up1, inputs);
      
      g[i] = (Cp1 - Cn1)/(2*h);

   return np.transpose(g);


def hessian(mpc_var, q, u0, u, inputs):
   # variable setup
   N = len(u);
   H = [[0 for i in range(N)] for j in range(N)];
   h = mpc_var.step_size;
   
   for i in range(N):
      un1 = [u[j] - (i==j)*h for j in range(N)];
      up1 = [u[j] + (i==j)*h for j in range(N)];
      
      gn1 = gradient(mpc_var, q, u0, un1, inputs);
      gp1 = gradient(mpc_var, q, u0, up1, inputs);
      
      H[i] = [(gp1[j] - gn1[j])/(2*h) for j in range(N)];

   return H;
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
