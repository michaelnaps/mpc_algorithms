import numpy as np
import cvxopt as co
import cvxopt.solvers as opt

def convert(id_var, q):
    # model variables
    m1_inputs = id_var.model1;
    N = m1_inputs.num_inputs;
    umax = m1_inputs.input_bounds;

    # state functions
    m1_dynamics = id_var.m1_dyn;
    m1_state    = id_var.m1_state;
    m1_jacobian = id_var.m1_jacob;

    # initial guess
    Z  = co.matrix([0 for i in range(N)]);
    Z3 = co.matrix([[Z], [Z], [Z]]);
    u_model1 = Z;

    (ddq, M, E) = m1_dynamics(q, u_model1, m1_inputs);
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
    dq_a = Ja*co.matrix(ddq[0:N]);

    # PID controller (temporary)
    kp = 100;  kd = 20;
    u_PD = kp*(q_a - q_d) + kd*(dq_a - dq_d);
    ud = dJa*dq_a - ddq_d + u_PD;

    # QP Optimization
    I = co.matrix([[float(i==j) for j in range(N)] for i in range(N)]);
    H = co.matrix([[Ja.T*Ja, Z3], [Z3, Z3]]);
    g = co.matrix([2*(Ja.T*ud), Z]);
    A = co.matrix([[M], [-I]], (3,6));
    b = E;

    # print(H, g, A, b)

    Aie = None;
    bie = None;

    lb = -3000*co.matrix(umax);
    ub =  3000*co.matrix(umax);

    u_model1 = opt.qp(H, g, Aie, bie, A, b)['x'][3:6];

    return [u_model1[i] for i in range(N)];
