import numpy as np
import cvxopt as co
import cvxopt.solvers as opt
from MathFunctionsCpp import MathExpressions

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
    Z  = co.matrix([0 for i in range(N+1)]);
    Z3 = co.matrix([[Z], [Z], [Z], [Z]]);
    u_model1 = [0, 0, 0];

    (_, M, E) = m1_dynamics(q, u_model1, m1_inputs);
    q_a = m1_state(q, u_model1, m1_inputs);
    (Ja, dJa) = m1_jacobian(q, m1_inputs);

    # convert matrices to cvxopt matrix type
    M  = co.matrix(M);     E   = co.matrix(E);
    Ja = co.matrix(Ja).T;  dJa = co.matrix(dJa).T;

    # desired state variables
    q_d   = co.matrix(id_var.m2_desired);
    dq_d  = Z;
    ddq_d = Z;

    # actual state variables
    q_a  = co.matrix(q_a);
    dq_a = Ja*co.matrix(q[N:2*N]);

    # centroidal momentum calculations
    np_q = np.array(q[0:N]);
    np_dq = np.array(q[N:2*N]);
    L    = co.matrix(mathexp.centroidal_momentum(np_q, np_dq));
    J_L  = co.matrix(mathexp.J_centroidal_momentum(np_q));
    dJ_L = co.matrix(mathexp.dJ_centroidal_momentum(np_q, np_dq));

    # include centroidal momentum in state vectors
    q_a = co.matrix([q_a, L]);
    dq_a = co.matrix([dq_a, 0]);
    J_a = co.matrix([Ja, J_L.T]);
    dJ_a = co.matrix([dJa, dJ_L]);

    print(q_a);  print(dq_a);
    print(q_d);  print(dq_d);
    print(dJ_a);  print(ddq_d);

    # PD controller (temporary)
    kp = co.matrix(np.diag([200, 50, 100, 20]));
    kd = co.matrix(np.diag([20, 20, 0, 0]));
    u_PD = kp*(q_a - q_d) + kd*(dq_a - dq_d);
    print(u_PD);  print(dJ_a*dq_a);
    u = dJ_a*dq_a - ddq_d + u_PD;

    # QP Optimization
    I = co.matrix([[float(i==j) for j in range(N)] for i in range(N)]);
    H = co.matrix([[J_a.T*J_a, Z3], [Z3, Z3]]);
    g = co.matrix([2*(J_a.T*u), Z]);
    A = co.matrix([[M], [-I]], (3,6));
    b = E;

    # print(H, g, A, b)

    Aie = None;
    bie = None;

    lb = -3000*co.matrix(umax);
    ub =  3000*co.matrix(umax);

    u_model1 = opt.qp(H, g, Aie, bie, A, b)['x'][3:6];

    return [u_model1[i] for i in range(N)];
