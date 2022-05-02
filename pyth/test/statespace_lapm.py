import numpy as np
import matplotlib.pyplot as plt

def statespace_lapm(q, u, inputs):
   # constant variables
   m = inputs.joint_masses[0];
   H = inputs.link_lengths[0];
   g = inputs.gravity_acc;
   
   # natural frequency
   w = np.sqrt(np.abs(g)/H);
   
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
   
   # capture point variables
   CP  = x + dx/w;
   CMP = H - H/g*ddq[0];
   
   # capture point dynamics (x)
   dCMP = w*(CP - CMP);

   return [dx, dLc, ddq[0], ddq[1], dCMP, 0];

# LAPM animation function
def animation_lapm(T, q, inputs):
   # sim variables
   Nt = len(T);
   dt = T[1] - T[0];
   
   # constant parameters
   H = inputs.link_lengths[0];
   L = inputs.link_lengths[0];
   d = inputs.CP_maxdistance;
   m = inputs.joint_masses[0];
   
   # limits of plot axes
   yLimits = [-0.1, L+L/10];
   xLimits = [-L/3, L/3];
   
   for i in range(Nt):
      plt.clf();
      
      x  = q[i][0];
      th = np.arccos(x/H);

      x0 = 0; y0 = 0;

      xlapm = x0 + x;
      ylapm = y0 + H;
      
      xtrue = x0 + L*np.cos(th);
      ytrue = y0 + L*np.sin(th);
      
      plt.plot([x0, xlapm], [y0, ylapm], linewidth=2, label="CoM - LAPM");
      plt.plot([x0, xtrue], [y0, ytrue], linestyle=':', label="CoM - True");
      plt.plot([xlapm], [ylapm], marker='o', 
         markerfacecolor='k', markeredgecolor='k', markersize=m/10);
      
      plt.plot([x0-L/4, x0+L/4], [y0, y0], 'k');
      plt.plot([xlapm, xlapm], [y0-d/2, y0+d/2], color='g', label="CP - Current");
      plt.plot([x0-d, x0-d], [y0-d/2, y0+d/2], 'k', label="CP - Bounds");
      plt.plot([x0+d, x0+d], [y0-d/2, y0+d/2], 'k');
      
      plt.title("TPM Simulation: t = {:.3f}".format(T[i]));
      plt.ylim(yLimits);  plt.xlim(xLimits);
      plt.legend();
      plt.grid();
      plt.pause(dt);
   
   input("Press enter to close animation...");

   return 1;
   
def plotStates_lapm(T, q):
   qT = np.transpose(q);
   
   fig, statePlot = plt.subplots(1,3);
   
   # CoM x-distance
   statePlot[0].plot(T, qT[0], label="x");
   statePlot[0].plot(T, qT[2], label="dx");
   statePlot[0].set_title("Position (x)");
   statePlot[0].legend();
   
   # angular momentum
   statePlot[1].plot(T, qT[1], label="Lc");
   statePlot[1].plot(T, qT[3], label="dLc");
   statePlot[1].set_title("Angular Momentum (L)");
   statePlot[1].legend();
   
   # capture point dynamics
   statePlot[2].plot(T, qT[4], label="CP");
   statePlot[2].set_title("Capture Point Dynamics (CP)");
   statePlot[2].legend();

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
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
