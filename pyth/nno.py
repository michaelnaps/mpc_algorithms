# Package: nno.py
# Created by: Michael Napoli
# Created on: 3/27/2022
# Last modified on: 3/27/2022


import numpy as np
import math
from modeuler import *


def mpc_root(mpc_var, model, q0, inputs):
   u0 = 0;
   u = newtons(mpc_var, model, q0, u0, inputs);
   return u;
   
   
def newtons(mpc_var, model, q0, u0, inputs):
   # variable setup
   N = len(q0)/2;
   eps = mpc_var.appx_zero;
   uc = u0;
   Cc = cost(mpc_var, model, q0, uc, inputs);
   un = uc;  Cn = Cc;
   
   count = 1;
   brk = 0;
   while(Cc != 0):
      g = gradient(mpc_var, model, q0, uc, inputs);
      gnorm = np.sqrt(np.sum(1))/N;
      
      if (gnorm < eps):
         brk = 1;
         break;
      
      H = hessian(mpc_var, model, q0, uc, inputs);
      un = uc - np.linalg.lstsq(H, g);
      Cn = cost(mpc_var, model, q0, un, inputs);
      count += 1;
      
      # break conditions
      if (count == 100):
         brk = -1;
         break;
         
      uc = un;  Cc = Cn;
   
   # while (Cc != 0):
   return u;
   

def cost(mpc_var, model, q0, u, inputs):
   # Cost of Constant Input
   # calculate the state over the desired prediction horizon
   qc = modeuler(mpc_var, model, q0, u, inputs);

   # sum of cost of each step of the prediction horizon
   du = (u0 - u);
   C = zeros(size(u));
   for i in range(P+1):
      C = C + Cq(thd, qc[i], du);
   end
   Cs = sum(C);
   

def gradient(mpc_var, model, q0, u, inputs):
   return 1;
   
def hessian(mpc_var, model, q0, u, inputs):
   return 1;
