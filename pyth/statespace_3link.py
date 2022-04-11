import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as anim

def statespace_3link(q, u, inputs):
   # Constants and State Variables
   N = inputs.num_joints;
   g = inputs.gravity_acc;
   c = inputs.damping_coefficients;
   m = inputs.joint_masses;
   L = inputs.link_lengths;
   
   m1 = m[0];      m2 = m[1];      m3 = m[2];
   L1 = L[0];      L2 = L[1];      L3 = L[2];
   r1 = L1/2;      r2 = L2/2;      r3 = L3/2;
   I1 = m1*L1/12;  I2 = m2*L2/12;  I3 = m3*L3/12;
   
   q1 = q[0];  q2 = q[1];  q3 = q[2];
   q4 = q[3];  q5 = q[4];  q6 = q[5];

   u1 = u[0];  u2 = u[1];  u3 = 0;  # u3 = u[2];
   c1 = c[0];  c2 = c[1];  c3 = c[2];

   # State Space Equations
   # Equation: E*ddq = M (rearrange for ddq)
   M = [[0 for j in range(N)] for i in range(N)]
   M[0][0] = -I3 - m3*r3**2 - L1*m3*r3*np.cos(q2 + q3) - L2*m3*r3*np.cos(q3);
   M[0][1] = -m3*r3**2 - L2*m3*np.cos(q3)*r3 - I3;
   M[0][2] = -m3*r3**2 - I3;
   M[1][0] = -m3*L2**2 - 2*m3*np.cos(q3)*L2*r3 - L1*m3*np.cos(q2)*L2 - m2*r2**2 - L1*m2*np.cos(q2)*r2 - m3*r3**2 - L1*m3*np.cos(q2 + q3)*r3 - I2 - I3;
   M[1][1] = -m3*L2**2 - 2*m3*np.cos(q3)*L2*r3 - m2*r2**2 - m3*r3**2 - I2 - I3;
   M[1][2] = -m3*r3**2 - L2*m3*np.cos(q3)*r3 - I3;
   M[2][0] = -I1 - I2 - I3 - L1**2*m2 - L1**2*m3 - L2**2*m3 - m1*r1**2 - m2*r2**2 - m3*r3**2 - 2*L1*m3*r3*np.cos(q2 + q3) - 2*L1*L2*m3*np.cos(q2) - 2*L1*m2*r2*np.cos(q2) - 2*L2*m3*r3*np.cos(q3);
   M[2][1] = -m3*L2**2 - 2*m3*np.cos(q3)*L2*r3 - L1*m3*np.cos(q2)*L2 - m2*r2**2 - L1*m2*np.cos(q2)*r2 - m3*r3**2 - L1*m3*np.cos(q2 + q3)*r3 - I2 - I3;
   M[2][2] = -I3 - m3*r3**2 - L1*m3*r3*np.cos(q2 + q3) - L2*m3*r3*np.cos(q3);

   E = [
      g*m3*r3*np.cos(q1 + q2 + q3) + c3*L3*q6 - u3 + L1*m3*r3*q4**2*np.sin(q2 + q3) + L2*m3*r3*q4**2*np.sin(q3) + L2*m3*r3*q5**2*np.sin(q3) + 2*L2*m3*r3*q4*q5*np.sin(q3),
      L2*g*m3*np.cos(q1 + q2) + c2*L2*q5 - u2 + g*m2*r2*np.cos(q1 + q2) + g*m3*r3*np.cos(q1 + q2 + q3) + L1*m3*r3*q4**2*np.sin(q2 + q3) + L1*L2*m3*q4**2*np.sin(q2) + L1*m2*r2*q4**2*np.sin(q2) - L2*m3*r3*q6**2*np.sin(q3) - 2*L2*m3*r3*q4*q6*np.sin(q3) - 2*L2*m3*r3*q5*q6*np.sin(q3),
      L2*g*m3*np.cos(q1 + q2) + c1*L1*q4 - u1 + g*m2*r2*np.cos(q1 + q2) + L1*g*m2*np.cos(q1) + L1*g*m3*np.cos(q1) + g*m1*r1*np.cos(q1) + g*m3*r3*np.cos(q1 + q2 + q3) - L1*m3*r3*q5**2*np.sin(q2 + q3) - L1*m3*r3*q6**2*np.sin(q2 + q3) - L1*L2*m3*q5**2*np.sin(q2) - L1*m2*r2*q5**2*np.sin(q2) - L2*m3*r3*q6**2*np.sin(q3) - 2*L1*m3*r3*q4*q5*np.sin(q2 + q3) - 2*L1*m3*r3*q4*q6*np.sin(q2 + q3) - 2*L1*m3*r3*q5*q6*np.sin(q2 + q3) - 2*L1*L2*m3*q4*q5*np.sin(q2) - 2*L1*m2*r2*q4*q5*np.sin(q2) - 2*L2*m3*r3*q4*q6*np.sin(q3) - 2*L2*m3*r3*q5*q6*np.sin(q3)
   ];

   dq = np.linalg.solve(M,E);

   return [q[3], q[4], q[5], dq[0], dq[1], dq[2]];


def animation_3link(T, q, inputs):
   Nt = len(T);
   dt = T[1] - T[0];
   
   L1 = inputs.link_lengths[0];
   L2 = inputs.link_lengths[1];
   L3 = inputs.link_lengths[2];
   
   axesLimits = [-(L1+L2+L3+0.5), (L1+L2+L3+0.5)];   
   
   for i in range(Nt):
      plt.clf();
      
      th1 = q[i][0];
      th2 = q[i][1];
      th3 = q[i][2];
      
      xAnkle = 0;                              yAnkle = 0;
      xKnee  = xAnkle + L1*np.cos(th1);        yKnee  = yAnkle + L1*np.sin(th1);
      xHip   = xKnee + L2*np.cos(th1+th2);     yHip   = yKnee + L2*np.sin(th1+th2);
      xHead  = xHip + L3*np.cos(th1+th2+th3);  yHead  = yHip + L3*np.sin(th1+th2+th3);
      
      plt.plot([xAnkle, xKnee], [yAnkle, yKnee]);
      plt.plot([xKnee, xHip], [yKnee, yHip]);
      plt.plot([xHip, xHead], [yHip, yHead]);
      
      plt.title("TPM Simulation: t = {:.3f}".format(T[i]));
      plt.xlim(axesLimits);
      plt.ylim(axesLimits);
      plt.grid();
      plt.pause(dt);
   
   input("Press Enter to close animation...");
   
   return 1;


def plotStates_3link(T, q):
   qT = np.transpose(q);
   
   fig, statePlot = plt.subplots(2,3);
   
   statePlot[0,0].plot(T, qT[0]);  statePlot[1,0].plot(T, qT[3]);
   statePlot[0,1].plot(T, qT[1]);  statePlot[1,1].plot(T, qT[4]);
   statePlot[0,2].plot(T, qT[2]);  statePlot[1,2].plot(T, qT[5]);

   return statePlot;
   
def plotInputs_3link(T, u):
   uT = np.transpose(u);
   
   fig, inputPlot = plt.subplots(1,3);
   
   inputPlot[0].plot(T, uT[0]);
   inputPlot[1].plot(T, uT[1]);
   inputPlot[2].plot(T, uT[2]);
   
   return inputPlot;
   
def plotCost_3link(T, C):
	fig, costPlot = plt.subplots();
	costPlot.plot(T, C);
	
	return costPlot;
