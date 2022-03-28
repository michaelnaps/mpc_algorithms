# File: modeuler.py
# Created by: Michael Napoli
# Created on: 1/17/2022
#
# Purpose: To calculate the states of a given statespace
#   over a predetermined prediction horizon.

import math
from statespace_3link import *

def modeuler(P, dt, q0, u, statespace):
   dt_min = 1e-3;
   if (dt >= dt_min):
      adj = int(dt/dt_min);

   jNum = len(u);
   Pm = P*adj;  dtm = dt/adj;
   q = [[0 for j in range(2*jNum)] for i in range(P+1)];
   qm = [[0 for j in range(2*jNum)] for i in range(Pm+1)];

   q[0] = q0;
   qm[0] = q0;
   for i in range(Pm):
      dq1 = statespace(qm[i], u,);
      qeu = [(qm[i][j] + dq1[j]*dtm) for j in range(2*jNum)];
      dq2 = statespace(qeu, u,);
      qm[i+1] = [(qm[i][j] + 1/2*(dq1[j] + dq2[j])*dtm) for j in range(2*jNum)];

      if (i % adj == 0):
         q[int(i/adj+1)] = qm[i+1];

   return q;

P = 4;
dt = 0.025;
q0 = [math.pi/2, 0, math.pi, 0, math.pi, 0];
u = [0, 0, 0];

qc = modeuler(P, dt, q0, u, statespace_3link);
for i in range(P+1):
   print(qc[i]);
