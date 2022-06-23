import numpy as np
from MathFunctionsCpp import MathExpressions
import time
import matplotlib.pyplot as plt

mathexp = MathExpressions()

def statespace_3link(q, u, inputs):  

    M = MassMatrix_3link(q, inputs);
    C = DriftVector_3link(q);

    E = [C[i] + u[i] for i in range(3)];

    dq = np.linalg.solve(M, E);

    return [q[3], q[4], q[5], dq[0], dq[1], dq[2]];

def MassMatrix_3link(q, inputs):
    # Constants and State Variables
    N = inputs.num_inputs;
    g = inputs.gravity_acc;
    c = inputs.damping_coefficients;
    m = inputs.joint_masses;
    L = inputs.link_lengths;

    m1 = m[0];      m2 = m[1];      m3 = m[2];
    L1 = L[0];      L2 = L[1];      L3 = L[2];
    r1 = L1/2;      r2 = L2/2;      r3 = L3/2;
    I1 = m1*L1/12;  I2 = m2*L2/12;  I3 = m3*L3/12;

    q1 = q[0];  q2 = q[1];  q3 = q[2];

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

    Mdq = [M[2], M[1], M[0]];
    Mdq = [[-Mdq[i][j] for j in range(N)] for i in range(N)];

    return Mdq;

def DriftVector_3link(q):
    mathexp = MathExpressions();

    q1 = q[0];  q2 = q[1];  q3 = q[2];
    q4 = q[3];  q5 = q[4];  q6 = q[5];

    x = np.array([q1,q2,q3]).reshape(3,1);
    dx = np.array([q4,q5,q6]).reshape(3,1);

    C = np.zeros(3);
    C = C + mathexp.Cmat_1(x, dx);
    C = C + mathexp.Cmat_2(x, dx);
    C = C + mathexp.Cmat_3(x, dx);
    C = np.matmul(C, dx);
    G = mathexp.Ge_vec(x);
    E = (C + G);

    return [E[i][0] for i in range(3)];

def CoM_3link(q, inputs):
    # Constants and State Variables
    N = inputs.num_inputs;
    m = inputs.joint_masses;
    L = inputs.link_lengths;

    m1 = m[0];  m2 = m[1];  m3 = m[2];
    L1 = L[0];  L2 = L[1];  L3 = L[2];
    r1 = L1/2;  r2 = L2/2;  r3 = L3/2;
    q1 = q[0];  q2 = q[1];  q3 = q[2];

    # compute center of mass components
    x1 = 0;  z1 = 0;
    x2 = x1 + L1*np.cos(q1);
    z2 = z1 + L1*np.sin(q1);
    x3 = x2 + L2*np.cos(q1 + q2);
    z3 = z2 + L2*np.sin(q1 + q2);

    x_com1 = x1 + r1*np.cos(q1);
    z_com1 = z1 + r1*np.sin(q1);
    x_com2 = x2 + r2*np.cos(q1 + q2);
    z_com2 = z2 + r2*np.sin(q1 + q2);
    x_com3 = x3 + r3*np.cos(q1 + q2 + q3);
    z_com3 = z3 + r3*np.sin(q1 + q2 + q3);

    x_com = (m1*x_com1 + m2*x_com2 + m3*x_com3)/(m1 + m2 + m3);
    z_com = (m1*z_com1 + m2*z_com2 + m3*z_com3)/(m1 + m2 + m3);
    q_com = q1 + q2 + q3;

    return (x_com, z_com, q_com);
    #return (z_com, q_com);

def J_CoM_3link(q, inputs):
    # Constants and State Variables
    N = inputs.num_inputs;
    g = inputs.gravity_acc;
    c = inputs.damping_coefficients;
    m = inputs.joint_masses;
    L = inputs.link_lengths;

    m1 = m[0];  m2 = m[1];  m3 = m[2];
    L1 = L[0];  L2 = L[1];  L3 = L[2];
    r1 = L1/2;  r2 = L2/2;  r3 = L3/2;

    q1 = q[0];  q2 = q[1];  q3 = q[2];
    q4 = q[3];  q5 = q[4];  q6 = q[5];

    # compute jacobian variables
    J_x_com = [
        -(m3*(r3*np.sin(q1 + q2 + q3) + L2*np.sin(q1 + q2) + L1*np.sin(q1)) + m2*(r2*np.sin(q1 + q2) + L1*np.sin(q1)) + m1*r1*np.sin(q1))/(m1 + m2 + m3),
        -(m3*(r3*np.sin(q1 + q2 + q3) + L2*np.sin(q1 + q2)) + m2*r2*np.sin(q1 + q2))/(m1 + m2 + m3),
        -(m3*r3*np.sin(q1 + q2 + q3))/(m1 + m2 + m3)
    ];

    J_z_com = [
        (m2*(r2*np.cos(q1 + q2) + L1*np.cos(q1)) + m3*(L2*np.cos(q1 + q2) + L1*np.cos(q1) + r3*np.cos(q1 + q2 + q3)) + m1*r1*np.cos(q1))/(m1 + m2 + m3),
        (m3*(L2*np.cos(q1 + q2) + r3*np.cos(q1 + q2 + q3)) + m2*r2*np.cos(q1 + q2))/(m1 + m2 + m3),
        (m3*r3*np.cos(q1 + q2 + q3))/(m1 + m2 + m3)
    ];

    J_q_com = [1, 1, 1];

    # compute derivative of jacobian components
    dJ_x_com = [
        -(q4*(m2*(r2*np.cos(q1 + q2) + L1*np.cos(q1)) + m3*(L2*np.cos(q1 + q2) + L1*np.cos(q1) + r3*np.cos(q1 + q2 + q3)) + m1*r1*np.cos(q1)))/(m1 + m2 + m3) - (q5*(m3*(L2*np.cos(q1 + q2) + r3*np.cos(q1 + q2 + q3)) + m2*r2*np.cos(q1 + q2)))/(m1 + m2 + m3) - (q6*m3*r3*np.cos(q1 + q2 + q3))/(m1 + m2 + m3),
        -(q4*(m3*(L2*np.cos(q1 + q2) + r3*np.cos(q1 + q2 + q3)) + m2*r2*np.cos(q1 + q2)))/(m1 + m2 + m3) - (q5*(m3*(L2*np.cos(q1 + q2) + r3*np.cos(q1 + q2 + q3)) + m2*r2*np.cos(q1 + q2)))/(m1 + m2 + m3) - (q6*m3*r3*np.cos(q1 + q2 + q3))/(m1 + m2 + m3),
        -(q4*m3*r3*np.cos(q1 + q2 + q3))/(m1 + m2 + m3) - (q5*m3*r3*np.cos(q1 + q2 + q3))/(m1 + m2 + m3) - (q6*m3*r3*np.cos(q1 + q2 + q3))/(m1 + m2 + m3)
    ];

    dJ_z_com = [
        -(q5*(m3*(r3*np.sin(q1 + q2 + q3) + L2*np.sin(q1 + q2)) + m2*r2*np.sin(q1 + q2)))/(m1 + m2 + m3) - (q4*(m3*(r3*np.sin(q1 + q2 + q3) + L2*np.sin(q1 + q2) + L1*np.sin(q1)) + m2*(r2*np.sin(q1 + q2) + L1*np.sin(q1)) + m1*r1*np.sin(q1)))/(m1 + m2 + m3) - (q6*m3*r3*np.sin(q1 + q2 + q3))/(m1 + m2 + m3),
        -(q4*(m3*(r3*np.sin(q1 + q2 + q3) + L2*np.sin(q1 + q2)) + m2*r2*np.sin(q1 + q2)))/(m1 + m2 + m3) - (q5*(m3*(r3*np.sin(q1 + q2 + q3) + L2*np.sin(q1 + q2)) + m2*r2*np.sin(q1 + q2)))/(m1 + m2 + m3) - (q6*m3*r3*np.sin(q1 + q2 + q3))/(m1 + m2 + m3),
        -(q4*m3*r3*np.sin(q1 + q2 + q3))/(m1 + m2 + m3) - (q5*m3*r3*np.sin(q1 + q2 + q3))/(m1 + m2 + m3) - (q6*m3*r3*np.sin(q1 + q2 + q3))/(m1 + m2 + m3)
    ];

    dJ_q_com = [0, 0, 0];

    J  = [J_x_com, J_z_com, J_q_com];
    dJ = [dJ_x_com, dJ_z_com, dJ_q_com];

    #J  = [J_z_com, J_q_com];
    #dJ = [dJ_z_com, dJ_q_com];

    return (J, dJ);

def animation_3link(T, q, inputs, save=0, filename="image.png"):
    Nt = len(T);
    dt = T[1] - T[0];

    L1 = inputs.link_lengths[0];
    L2 = inputs.link_lengths[1];
    L3 = inputs.link_lengths[2];

    m1 = inputs.joint_masses[0];
    m2 = inputs.joint_masses[1];
    m3 = inputs.joint_masses[2];
    m  = m1 + m2 + m3;

    if save:
        xLimits = [-(L1+L2+L3)/2+0.6, (L1+L2+L3)/2];
        yLimits = [-(L1+L2+L3)/4+0.3, (L1+L2+L3)*0.75];
    else:
        xLimits = [-(L1+L2+L3)/2, (L1+L2+L3)/2];
        yLimits = [-(L1+L2+L3)/2, (L1+L2+L3)];

    plt.figure(figsize=(4,5.2), dpi=80);

    for i in range(Nt):
        plt.clf();

        q1 = q[i][0];
        q2 = q[i][1];
        q3 = q[i][2];

        xAnkle = 0;                           yAnkle = 0;
        xKnee  = xAnkle + L1*np.cos(q1);      yKnee  = yAnkle + L1*np.sin(q1);
        xHip   = xKnee + L2*np.cos(q1+q2);    yHip   = yKnee + L2*np.sin(q1+q2);
        xHead  = xHip + L3*np.cos(q1+q2+q3);  yHead  = yHip + L3*np.sin(q1+q2+q3);

        (x_com, y_com, _) = CoM_3link(q[i], inputs);

        plt.plot([xAnkle, xKnee], [yAnkle, yKnee]);
        plt.plot([xKnee, xHip], [yKnee, yHip]);
        plt.plot([xHip, xHead], [yHip, yHead]);
        plt.plot([xKnee], [yKnee], linestyle="none",
                 marker='.', markerfacecolor='k', markeredgecolor='k', markersize=m1, label="Joint Mass");
        plt.plot([xHip], [yHip], linestyle="none",
                  marker='.', markerfacecolor='k', markeredgecolor='k', markersize=m2);
        plt.plot([xHead], [yHead], linestyle="none",
                  marker='.', markerfacecolor='k', markeredgecolor='k', markersize=m3);
        plt.plot([x_com], [y_com], linestyle="none",
                  marker='s', markerfacecolor='m', markeredgecolor='m', markersize=m/10, label="COM");
        plt.legend();

        plt.title("TPM Simulation: t = {:.3f}".format(T[i]));
        plt.xticks(color='w');  plt.yticks(color='w');
        plt.xlim(xLimits);  plt.ylim(yLimits);
        plt.grid();
        plt.pause(dt);

    if save:
        plt.savefig(filename, format="png", dpi=800);

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




def calcMassMatrix(q):
    #  calculates the mass matrix Mmat(x)
    
    #  Parameters:
    #  q: the state variables @type colvec

    MmatName_= ["Mmat_L1_pend_3link","Mmat_L2_pend_3link","Mmat_L3_pend_3link"]
    Dimension = 3

    Mmat_name = MmatName_
    n_fun = len(Mmat_name)
    M = np.zeros((Dimension,Dimension))

    # for i in range(n_fun):
    #     M = M + eval(Mmat_name[i],q)
    

    M = np.zeros((Dimension,Dimension))
    M = M + mathexp.Mmat_L1_pend_3link(q)
    M = M + mathexp.Mmat_L2_pend_3link(q)
    M = M + mathexp.Mmat_L3_pend_3link(q)

    return M




