import numpy as np
import cvxopt as co
import cvxopt.solvers as opt
from MathFunctionsCpp import MathExpressions
import math

def convert(id_var, q):
    mathexp = MathExpressions();

    # model variables
    m1_inputs = id_var.model1;
    N = m1_inputs.num_inputs;
    umax = m1_inputs.input_bounds;

    # state functions
    m1_dynamics = id_var.m1_dyn;
    m1_state    = id_var.m1_state;
    m1_jacobian = id_var.m1_jacob;

    # initial guess and zero matrices
    Z  = co.matrix([0 for i in range(N)]);
    Z3 = co.matrix([[Z], [Z], [Z]]);
    I = co.matrix([[float(i==j) for j in range(N)] for i in range(N)]);
    u_model1 = [0, 0, 0];

    # current joint states
    x  = co.matrix(q[:N]);
    dx = co.matrix(q[N:2*N]);

    print("x =", x);  print("dx =", dx);

    (_, M, E) = m1_dynamics(q, u_model1, m1_inputs);  print(E);
    q_a = m1_state(q, m1_inputs);
    (Ja, dJa) = m1_jacobian(q, m1_inputs);

    # convert matrices to cvxopt matrix type
    M = co.matrix(M);  J_a  = co.matrix(Ja).T;
    E = co.matrix(E);  dJ_a = co.matrix(dJa).T;

    print("J_a =\n", J_a);  print("dJ_a =\n", dJ_a);

    # desired state variables
    q_d   = co.matrix(id_var.m2_desired)[:N];
    L_d   = id_var.m2_desired[N];
    dq_d  = Z;
    ddq_d = Z;

    # actual state variables
    q_a  = co.matrix(q_a);
    dq_a = J_a*co.matrix(q[N:2*N]);

    print("q_a =", q_a);  print("dq_a =", dq_a);

    # centroidal momentum calculations
    np_x = np.array(x);
    np_dx = np.array(dx);
    L    = co.matrix(mathexp.centroidal_momentum(np_x, np_dx));
    J_L  = co.matrix(mathexp.J_centroidal_momentum(np_x));
    dJ_L = co.matrix(mathexp.dJ_centroidal_momentum(np_x, np_dx));

    # PD controller (temporary)
    kp = co.matrix(np.diag([200, 50, 100]));
    kd = co.matrix(np.diag([20, 20, 0]));
    u_PD = kp*(q_a - q_d) + kd*(dq_a - dq_d);

    u_q = dJ_a*dx - ddq_d + u_PD;
    u_L = dJ_L*dx + 20*(L - L_d);

    J  = co.matrix([J_a, J_L.T]);
    dJ = co.matrix([dJ_a, dJ_L]);
    u  = co.matrix([u_q, u_L]);

    print("L =\n", L);  print("J_L =\n", J_L);  print("dJ_L =\n", dJ_L);

    print("J =\n", J);  print("dJ =\n", dJ);  print("u =\n", u)

    # QP Optimization
    H = co.matrix([[J.T*J, Z3], [Z3, Z3]]);
    g = co.matrix([2*(J.T*u), Z]);
    A = co.matrix([[M], [-I]], (3,6));
    b = E;

    print("H =\n", H);
    print("g =\n", g);
    print("A =\n", A);
    print("b =\n", b);

    # control barrier functions
    # Aie = None;
    # bie = None;
    gm1 = 2;  gm2 = 2;

    h_con = -co.matrix([
        q_a[0] + 0.1,
        -q_a[0] + 0.1,
        q_a[1] - 0.5,
        -q_a[1] + 1,
        q_a[2] - math.pi/2 + 1,
        -q_a[2] + math.pi/2 + 1
    ]);
    J_con = -co.matrix([
        J_a[0,:],
        -J_a[0,:],
        J_a[1,:],
        -J_a[1,:],
        J_a[2,:],
        -J_a[2,:],
    ]);
    dJ_con = -co.matrix([
        dJ_a[0,:],
        -dJ_a[0,:],
        dJ_a[1,:],
        -dJ_a[1,:],
        dJ_a[2,:],
        -dJ_a[2,:],
    ]);

    print("h_con =\n", h_con);
    print("J_con =\n", J_con);
    print("dJ_con =\n", dJ_con);

    Aie = co.matrix([[J_con], [Z3, Z3]]);
    bie = -(dJ_con + gm1*J_con)*dx - gm2*(J_con*dx + gm1*h_con);

    print("Aie =\n", Aie);  print("bie =\n", bie)

    lb = -3000*co.matrix(umax);
    ub =  3000*co.matrix(umax);

    u_model1 = opt.qp(H, g, Aie, bie, A, b)['x'][3:6];

    print("u_result =", u_model1);

    return [u_model1[i] for i in range(N)];
