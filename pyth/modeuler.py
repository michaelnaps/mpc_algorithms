# File: modeuler.py
# Created by: Michael Napoli
# Created on: 1/17/2022
#
# Purpose: To calculate the states of a given statespace
#   over a predetermined prediction horizon.
def modeuler(mpc_var, q0, u, inputs):
   P  = mpc_var.PH_length;
   dt = mpc_var.time_step;
   statespace = mpc_var.model;

   dt_min = 1e-3;
   if (dt >= dt_min):
      adj = int(dt/dt_min);

   jNum = len(u);
   Pm = P*adj;  dtm = dt/adj;
   q  = [[0 for j in range(2*jNum)] for i in range(P+1)];
   qm = [[0 for j in range(2*jNum)] for i in range(Pm+1)];

   q[0]  = q0;
   qm[0] = q0;
   for i in range(Pm):
      dq1 = statespace(qm[i], u, inputs);
      qeu = [(qm[i][j] + dq1[j]*dtm) for j in range(2*jNum)];
      dq2 = statespace(qeu, u, inputs);
      qm[i+1] = [(qm[i][j] + 1/2*(dq1[j] + dq2[j])*dtm) for j in range(2*jNum)];

      if (i % adj == 0):
         q[int(i/adj+1)] = qm[i+1];

   return q;
