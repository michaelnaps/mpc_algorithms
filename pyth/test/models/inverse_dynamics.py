import numpy as np
from qpsolvers import solve_qp
from MathFunctionsCpp import MathExpressions
import math
from statespace_tpm import *

def convert(inputs_tpm, q_desired, q, u_prev, output=0):
    mathexp = MathExpressions();

    # model variables
    N = inputs_tpm.num_inputs;

    # initial guess and zero matrices
    Z = np.array([0 for i in range(N)]);
    Z3 = np.array([[0 for j in range(N)] for i in range(N)]);
    I = np.array([[float(i==j) for j in range(N)] for i in range(N)]);

    # current joint states
    x  = np.array(q[:N]);     x.shape  = (N,1);
    dx = np.array(q[N:2*N]);  dx.shape = (N,1);

    if output:
        print("\nx =", x);  print("\ndx =", dx);

    # calculate the drift vector, mass matrix, and state jacobians
    M = np.array(MassMatrix_tpm(q, inputs_tpm));
    E = np.array(DriftVector_tpm(q))
    (J_a, dJ_a) = np.array(J_CoM_tpm(q, inputs_tpm));
    J_a = J_a[1:N];  dJ_a = dJ_a[1:N]

    if output:
        print("\nE =\n", E);  print("\nM =\n", M);
        print("\nJ_a =\n", J_a);  print("\ndJ_a =\n", dJ_a);

    # actual state variables
    q_a  = np.array(CoM_tpm(q, inputs_tpm))[1:N];  q_a.shape = (len(q_a),1);
    dq_a = np.matmul(J_a, dx);  dq_a.shape = (len(dq_a),1);

    if output:
        print("\nq_a =\n", q_a);
        print("\ndq_a =\n", dq_a);

    # desired state variables
    q_d   = np.array(q_desired)[1:N];
    q_d.shape = (len(q_d),1);
    Lc_d  = q_desired[-1];
    dq_d  = Z[1:N];  dq_d.shape  = (len(q_d),1);
    ddq_d = Z[1:N];  ddq_d.shape = (len(q_d),1);

    if output:
        print("\nq_d =\n", q_d);
        print("\nLc_d =\n", Lc_d);
        print("\ndq_d =\n", dq_d);

    # centroidal momentum calculations
    Lc    = mathexp.centroidal_momentum(x, dx)[0][0];
    J_Lc  = mathexp.J_centroidal_momentum(x);
    dJ_Lc = mathexp.dJ_centroidal_momentum(x, dx)[0];

    if output:
        print("\nLc =\n", Lc);
        print("\nJ_Lc =\n", J_Lc);
        print("\ndJ_Lc =\n", dJ_Lc);

    # PD controller (temporary)
    # kp = np.diag([7500, 10000, 1000]);
    # kd = np.diag([500, 1000, 1500]);
    kp = np.diag([10000, 1000]);
    kd = np.diag([1000, 1500]);
    u_PD = np.matmul(kp, (q_a - q_d)) + np.matmul(kd, (dq_a - dq_d));

    u_q  = np.matmul(dJ_a, dx) - ddq_d + u_PD;
    u_Lc = np.matmul(dJ_Lc, dx) + 15000*(Lc - Lc_d);

    if output:
        print("\nu_PD =\n", u_PD);
        print("\nu_q =\n", u_q);
        print("\nu_Lc =\n", u_Lc);

    J  = np.vstack((J_a, J_Lc.T));
    # J = J_a;
    dJ = np.vstack((dJ_a, dJ_Lc));
    # dJ = dJ_a;
    u  = np.append(u_q, u_Lc);  u.shape = (len(u), 1);
    # u = u_q;  u.shape = (len(u), 1);

    if output:
        print("\nJ =\n", J);
        print("\ndJ =\n", dJ);
        print("\nu =\n", u);

    # QP Optimization
    ku = 10;
    H = np.vstack((np.append(np.matmul(J.transpose(), J), Z3, axis=1), np.append(Z3, ku*I, axis=1)));
    g = np.append(2*np.matmul(J.transpose(), u), np.zeros(3));  g.shape = (len(g),);
    A = np.append(M, -I, axis=1);
    b = -E;  b.shape = (len(b),);

    if output:
        print("\nH =\n", H);
        print("\ng =\n", g);
        print("\nA =\n", A);
        print("\nb =\n", b);

    # control barrier functions
    gm1 = 10;  gm2 = 10;

    h_con = -np.array([
        # q_a[0] + 0.1,
        # -q_a[0] + 0.1,
        q_a[0] - 0.5,
        -q_a[0] + 1,
        q_a[1] - math.pi/2 + 1,
        -q_a[1] + math.pi/2 + 1
    ]);  h_con.shape = (len(h_con),1);
    J_con = -np.array([
        J_a[0],
        -J_a[0],
        J_a[1],
        -J_a[1]
        # J_a[2],
        # -J_a[2]
    ]);
    dJ_con = -np.array([
        dJ_a[0],
        -dJ_a[0],
        dJ_a[1],
        -dJ_a[1]
        # dJ_a[2],
        # -dJ_a[2]
    ]);

    if output:
        print("\nh_con =\n", h_con);
        print("\nJ_con =\n", J_con);
        print("\ndJ_con =\n", dJ_con);

    lb = -np.array([2000, 2000, 2000, 10000, 10000, 10000]);
    lb.shape = (len(lb),);
    ub = -lb;

    if output:
        print("\nlb =\n", lb);
        print("\nub =\n", ub)

    G = np.append(J_con, np.zeros(4*3).reshape(4,3), axis=1);
    h = -np.matmul((dJ_con + gm1*J_con), dx) - gm2*(np.matmul(J_con, dx) + gm1*h_con);
    h.shape = (len(h),)

    if output:
        print("\nG =\n", G);
        print("\nh =\n", h)

        print("\nMatrices Dimensions:");
        print("H.shape =", H.shape);
        print("g.shape =", g.shape);
        print("G.shape =", G.shape);
        print("h.shape =", h.shape);
        print("A.shape =", A.shape);
        print("b.shape =", b.shape);
        print("lb.shape =", lb.shape);
        print("ub.shape =", ub.shape);

    # G = None;
    # h = None;
    u_result = solve_qp(H, g, G, h, A, b, lb, ub, solver='cvxopt');

    if (u_result is None):
        return u_result;

    if output:
        print("\nu_result =", u_result);

    return [u_result[i] for i in range(N,2*N)];
