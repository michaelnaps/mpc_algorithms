# Package: nno.py
# Created by: Michael Napoli
# Created on: 3/27/2022
# Last modified on: 3/27/2022


import numpy as np
import math
from modeuler import *


def mpc_root(mpc_var, q0, inputs):
   u0 = 0;
   u = newtons(mpc_var, q0, u0, inputs);
   return u;
   
   
def newtons(mpc_var, q0, u0, inputs):
   # variable setup
   N = len(q0)/2;
   eps = mpc_var.appx_zero;
   uc = u0;
   Cc = cost(mpc_var, q0, uc, inputs);
   un = uc;  Cn = Cc;
   
   count = 1;
   brk = 0;
   while(Cc != 0):
      g = gradient(mpc_var, q0, uc, inputs);
      gnorm = np.sqrt(np.sum(1))/N;
      
      if (gnorm < eps):
         brk = 1;
         break;
      
      H = hessian(mpc_var, q0, uc, inputs);
      un = uc - np.linalg.lstsq(H, g);
      Cn = cost(mpc_var, q0, un, inputs);
      count += 1;
      
      # break conditions
      if (count == 100):
         brk = -1;
         break;
         
      uc = un;  Cc = Cn;
   
   # while (Cc != 0):
   return u;
   

def cost(mpc_var, q0, u, inputs):
   # MPC constants
   P  = mpc_var.PH_length;
   Cq = mpc_var.cost_func;
   qd = mpc_var.des_angles;
   
   # Cost of Constant Input
   # simulate over the prediction horizon and sum cost
   qc = modeuler(mpc_var, q0, u, inputs);

   C = [0 for i in range(len(u))];

   for i in range(P+1):
      C = C + Cq(qd, qc[i]);

   return np.sum(C);
   

def gradient(mpc_var, q, u, inputs):
   # variable setup
   N = len(u);
   g = [0 for i in range(N)];
   h = mpc_var.step_size;
   
   for i in range(N):
      un1 = [u[j] - (i==j)*h for j in range(N)];
      up1 = [u[j] + (i==j)*h for j in range(N)];
      
      Cn1 = cost(mpc_var, q, un1, inputs);
      Cp1 = cost(mpc_var, q, up1, inputs);
      
      g[i] = (Cp1 - Cn1)/(2*h);

   return np.transpose(g);


def hessian(mpc_var, q, u, inputs):
   # variable setup
   N = len(u);
   H = [[0 for i in range(N)] for j in range(N)];
   h = mpc_var.step_size;
   
   for i in range(N):
      un1 = [u[j] - (i==j)*h for j in range(N)];
      up1 = [u[j] + (i==j)*h for j in range(N)];
      
      gn1 = gradient(mpc_var, q, un1, inputs);
      gp1 = gradient(mpc_var, q, up1, inputs);
      
      H[i] = [(gp1[j] - gn1[j])/(2*h) for j in range(N)];

   return H;
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
