import numpy as np

def statespace_3link(q, u, c, m, L):
    # Constants and State Variables
    jNum = len(u);
    m1 = m[0];      m2 = m[1];      m3 = m[2];
    L1 = L[0];      L2 = L[1];      L3 = L[2];
    r1 = L1/2;      r2 = L2/2;      r3 = L3/2;
    I1 = m1*L1/12;  I2 = m2*L2/12;  I3 = m3*L3/12;
    g = 9.81;

    q1 = q[0];  q2 = q[2];  q3 = q[4];
    q4 = q[1];  q5 = q[3];  q6 = q[5];

    u1 = u[0]; u2 = u[1]; u3 = u[2];
    c1 = c[0]; c2 = c[1]; c3 = c[2];

    # State Space Equations
    # Equation: E*ddq = M (rearrange for ddq)
    M = [[0 for j in range(jNum)] for i in range(jNum)]
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

    return [q[1], dq[0], q[3], dq[1], q[5], dq[2]];
