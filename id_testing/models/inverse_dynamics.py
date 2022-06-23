import numpy as np
from qpsolvers import solve_qp
from MathFunctionsCpp import MathExpressions
import math
from statespace_3link import *

def convert(inputs_3link, q_desired, q, output=0):
    mathexp = MathExpressions();

    print(q);
    # model variables
    N = inputs_3link.num_inputs;

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
    M = np.array(MassMatrix_3link(q, inputs_3link));
    E = np.array(DriftVector_3link(q))
    (J_a, dJ_a) = np.array(J_CoM_3link(q, inputs_3link));
    J_a = J_a;  dJ_a = dJ_a;

    if output:
        print("\nE =\n", E);  print("\nM =\n", M);
        print("\nJ_a =\n", J_a);  print("\ndJ_a =\n", dJ_a);

    # actual state variables
    q_a  = np.array(CoM_3link(q, inputs_3link));  q_a.shape = (len(q_a),1);
    dq_a = np.matmul(J_a, dx);  dq_a.shape = (len(dq_a),1);

    if output:
        print("\nq_a =\n", q_a);
        print("\ndq_a =\n", dq_a);

    # desired state variables
    q_d   = np.array(q_desired)[:len(q_a)];
    q_d.shape = (len(q_d),1);
    Lc_d   = q_desired[-1];
    dq_d  = Z;  dq_d.shape  = (len(q_d),1);
    ddq_d = Z;  ddq_d.shape = (len(q_d),1);

    if output:
        print("\nq_d =\n", q_d);
        print("\nLc_d =\n", Lc_d);
        print("\ndq_d =\n", dq_d);

    print("\nq_a =\n", q_a);
    print("\nq_d =\n", q_d);

    # centroidal momentum calculations
    Lc    = mathexp.centroidal_momentum(x, dx);
    J_Lc  = mathexp.J_centroidal_momentum(x);
    dJ_Lc = mathexp.dJ_centroidal_momentum(x, dx)[0];

    # PD controller (temporary)
    kp = np.diag([400, 400, 400]);
    kd = np.diag([50, 50, 50]);
    u_PD = np.matmul(kp, (q_a - q_d)) + np.matmul(kd, (dq_a - dq_d));

    u_q = np.matmul(dJ_a, dx) - ddq_d + u_PD;
    u_Lc = 1*np.matmul(dJ_Lc, dx) + 160*(Lc - Lc_d);

    if output:
        print("\nu_PD =\n", u_PD);
        print("\nu_q =\n", u_q);
        print("\nu_Lc =\n", u_Lc);

    J  = np.vstack((J_a, J_Lc.T));
    dJ = np.vstack((dJ_a, dJ_Lc));
    u  = np.append(u_q, u_Lc);  u.shape = (len(u), 1);

    if output:
        print("\nLc =\n", Lc);
        print("\nJ_Lc =\n", J_Lc);
        print("\ndJ_Lc =\n", dJ_Lc);

        print("\nJ =\n", J);
        print("\ndJ =\n", dJ);
        print("\nu =\n", u);

    # QP Optimization
    H = np.vstack((np.append(np.matmul(J.transpose(), J), Z3, axis=1), np.append(Z3, Z3, axis=1)));
    g = np.append(2*np.matmul(J.transpose(), u), np.zeros(N));  g.shape = (len(g),);
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
        q_a[0] + 0.1,
        -q_a[0] + 0.1,
        q_a[1] - 0.5,
        -q_a[1] + 1,
        q_a[2] - math.pi/2 + 1,
        -q_a[2] + math.pi/2 + 1
    ]);  h_con.shape = (len(h_con),1);
    J_con = -np.array([
        J_a[0],
        -J_a[0],
        J_a[1],
        -J_a[1],
        J_a[2],
        -J_a[2]
    ]);
    dJ_con = -np.array([
        dJ_a[0],
        -dJ_a[0],
        dJ_a[1],
        -dJ_a[1],
        dJ_a[2],
        -dJ_a[2]
    ]);

    if output:
        print("\nh_con =\n", h_con);
        print("\nJ_con =\n", J_con);
        print("\ndJ_con =\n", dJ_con);

    lb = -np.array([2000, 2000, 2000, 40, 1000, 500]);
    lb.shape = (len(lb),);
    ub = -lb;

    if output:
        print("\nlb =\n", lb);
        print("\nub =\n", ub)

    G = np.append(J_con, np.zeros(6*3).reshape(6,3), axis=1);
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

    # G = None
    # h = None
    u_result = solve_qp(H, g, G, h, A, b, lb, ub, solver='cvxopt');

    if (u_result is None):
        return u_result;

    if output:
        print("\nu_result =", u_result);

    return [u_result[i] for i in range(N,2*N)], [u_result[i] for i in range(0,N)];
