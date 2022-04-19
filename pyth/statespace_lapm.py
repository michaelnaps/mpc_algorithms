import numpy as np
import matplotlib.pyplot as plt

def statespace_lapm(q, u, inputs):
   # constant variables
   m = inputs.joint_masses[0];
   H = inputs.link_lengths[0];
   g = inputs.gravity_acc;
   
   # state variables
   x  = q[0];  Lc  = q[1];
   dx = q[2];  dLc = q[3];
   
   # input variables
   ua = u[0];  L = u[1];
   
   # solve for statespace
   ddq = [
      L/(m*H) - Lc/(m*H),
      ua + m*g*x
   ];

   return [dx, dLc, ddq[0], ddq[1]];

# LAPM animation function
def animation_lapm(T, q, inputs):
   # sim variables
   Nt = len(T);
   dt = T[1] - T[0];
   
   # constant parameters
   H = inputs.link_lengths[0];
   L = inputs.link_lengths[0];
   
   # limits of plot axes
   axesLimits = [-L-1/4, L+1/4];
   
   for i in range(Nt):
      plt.clf();
      
      x  = q[i][0];
      th = np.arccos(x/H);

      x0 = 0; y0 = 0;

      xlapm = x0 + x;
      ylapm = H;
      
      xtrue = x0 + L*np.cos(th);
      ytrue = y0 + L*np.cos(th);

      plt.plot([x0, xlapm], [y0, ylapm]);
      plt.plot([x0, xtrue], [y0, ytrue]);
      
      plt.title("TPM Simulation: t = {:.3f}".format(T[i]));
      plt.ylim(axesLimits);  plt.xlim(axesLimits);
      plt.grid();
      plt.pause(dt);
   
   input("Press enter to close animation...");

   return 1;
   
def plotStates_lapm(T, q):
   qT = np.transpose(q);
   
   fig, statePlot = plt.subplots(2,2);
   
   statePlot[0,0].plot(T, qT[0]);  statePlot[1,0].plot(T, qT[2]);
   statePlot[0,1].plot(T, qT[1]);  statePlot[1,1].plot(T, qT[3]);

   return statePlot;
   
def plotInputs_lapm(T, u):
   uT = np.transpose(u);
   
   fig, inputPlot = plt.subplots(1,2);
   
   inputPlot[0].plot(T, uT[0]);
   inputPlot[1].plot(T, uT[1]);
   
   return inputPlot;
   
def plotCost_lapm(T, C):
	fig, costPlot = plt.subplots();
	costPlot.plot(T, C);
	
	return costPlot;
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
