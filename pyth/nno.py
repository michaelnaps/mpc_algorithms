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
   q = [[0 for i in range(N)] for j in range(Nt)];
   q[0] = q0;
   
   # loop input
   unew = u0;

   for i in range(1,Nt):
      unew = newtons(mpc_var, q[i-1], unew, unew, inputs)[0];
      qnew = modeuler(mpc_var, q[i-1], unew, inputs)[1];
      q[i] = qnew[0];
   
   return (T, q);
   
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
      g = gradient(mpc_var, q0, u0, uc, inputs);
      gnorm = np.sqrt(np.sum([g[i]**2 for i in range(N)]))/N;
      
      if (gnorm < eps):
         brk = 1;
         break;
      
      H = hessian(mpc_var, q0, u0, uc, inputs);
      udn = np.linalg.lstsq(H, g, rcond=None)[0];
      un = [uc[i] - udn[i] for i in range(N)];
      Cn = cost(mpc_var, q0, u0, un, inputs);
      count += 1;
      
      # break conditions
      if (count == 100):
         brk = -1;
         break;
         
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
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
