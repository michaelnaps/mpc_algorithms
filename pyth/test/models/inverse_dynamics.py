import numpy as np
# import cvxopt as co
# import cvxopt.solvers as opt
from qpsolvers import solve_qp
from MathFunctionsCpp import MathExpressions
import math

def convert(id_var, q_desired, q, u0, output=0):
    mathexp = MathExpressions();

    # model variables
    m1_inputs = id_var.model1;
    N = m1_inputs.num_inputs;

    # state functions
    m1_massMatrix = id_var.m1_mass;
    m1_state    = id_var.m1_state;
    m1_jacobian = id_var.m1_jacob;

    # initial guess and zero matrices
    Z = np.array([0 for i in range(N)]);
    Z3 = np.array([[0 for j in range(N)] for i in range(N)]);
    I = np.array([[float(i==j) for j in range(N)] for i in range(N)]);
    u_model1 = np.array(u0);

    # current joint states
    x  = np.array(q[:N]);     x.shape  = (N,1);
    dx = np.array(q[N:2*N]);  dx.shape = (N,1);

    if output:
        print("\nx =", x);  print("\ndx =", dx);

    M = np.array(m1_massMatrix(q, u_model1, m1_inputs));
    (J_a, dJ_a) = np.array(m1_jacobian(q, m1_inputs));
    J_a = J_a;  dJ_a = dJ_a;

    # calculate the drift vector
    C = Z3;
    C = C + mathexp.Cmat_1(x, dx);
    C = C + mathexp.Cmat_2(x, dx);
    C = C + mathexp.Cmat_3(x, dx);
    C = np.matmul(C, dx);
    G = mathexp.Ge_vec(x);
    E = -(C + G);

    if output:
        print("\nJ_a =\n", J_a);  print("\ndJ_a =\n", dJ_a);

    # actual state variables
    q_a  = np.array(m1_state(q, m1_inputs));
    q_a.shape = (len(q_a),1);
    dq_a = np.matmul(J_a, dx);
    dq_a.shape = (len(dq_a),1);

    if output:
        print("\nq_a =\n", q_a);
        print("\ndq_a =\n", dq_a);

    # desired state variables
    q_d   = np.array(q_desired)[:len(q_a)];
    q_d.shape = (len(q_d),1);
    L_d   = q_desired[len(q_a)];
    dq_d  = Z;  dq_d.shape  = (len(q_d),1);
    ddq_d = Z;  ddq_d.shape = (len(q_d),1);

    if output:
        print("\nq_d =\n", q_d);
        print("\nL_d =\n", L_d);
        print("\ndq_d =\n", dq_d);

    # centroidal momentum calculations
    L    = mathexp.centroidal_momentum(x, dx);
    J_L  = mathexp.J_centroidal_momentum(x);
    dJ_L = mathexp.dJ_centroidal_momentum(x, dx)[0];

    # PD controller (temporary)
    kp = np.diag([50, 50, 100]);
    kd = np.diag([20, 20, 0]);
    u_PD = np.matmul(kp, (q_a - q_d)) + np.matmul(kd, (dq_a - dq_d));

    u_q = np.matmul(dJ_a, dx) - ddq_d + u_PD;
    u_L = np.matmul(dJ_L, dx) + 20*(L - L_d);

    if output:
        print("\nu_PD =\n", u_PD);
        print("\nu_q =\n", u_q);
        print("\nu_L =\n", u_L);

    J  = np.vstack((J_a, J_L.T));
    dJ = np.vstack((dJ_a, dJ_L));
    u  = np.append(u_q, u_L);  u.shape = (len(u), 1);

    if output:
        print("\nL =\n", L);
        print("\nJ_L =\n", J_L);
        print("\ndJ_L =\n", dJ_L);

        print("\nJ =\n", J);
        print("\ndJ =\n", dJ);
        print("\nu =\n", u);

    # QP Optimization
    H = np.vstack((np.append(np.matmul(J.transpose(), J), Z3, axis=1), np.append(Z3, Z3, axis=1)));
    g = np.append(2*np.matmul(J.transpose(), u), np.zeros(N));  g.shape = (len(g),);
    A = np.append(M, -I, axis=1);
    b = E;  b.shape = (len(b),);

    if output:
        print("\nH =\n", H);
        print("\ng =\n", g);
        print("\nA =\n", A);
        print("\nb =\n", b);

    # control barrier functions
    gm1 = 2;  gm2 = 2;

    h_con = -np.array([
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
    ]);
    dJ_con = -np.array([
        dJ_a[0],
        -dJ_a[0],
        dJ_a[1],
        -dJ_a[1]
    ]);

    if output:
        print("\nh_con =\n", h_con);
        print("\nJ_con =\n", J_con);
        print("\ndJ_con =\n", dJ_con);

    lb = -np.array([2000, 2000, 2000, 1000, 1000, 1000]);
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

    u_model1 = solve_qp(H, g, G, h, A, b, lb, ub, solver='cvxopt')[N:2*N];

    if output:
        print("\nu_result =", u_model1);

    return [u_model1[i] for i in range(N)];
