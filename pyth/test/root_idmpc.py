import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import mpc
from statespace_alip import *
from statespace_3link import *
from modeuler import modeuler
from MathFunctionsCpp import MathExpressions
import inverse_dynamics as id
import math
import matplotlib.pyplot as plt

def Cq(qd, q):
    Cq = [
        100*(qd[0] - q[0])**2,
         10*(qd[1] - q[1])**2
    ];

    return np.sum(Cq);

def Cu(u, du, inputs):
    umax = inputs.input_bounds;

    Cu = [
        1e-5*(du[0])**2 - np.log(umax[0]**2 - u[0]**2) + np.log(umax[0]**2),
        1e-5*(du[1])**2# + (u[1]/umax[1])**2 #- np.log(umax[1]**2 - u[1]**2) + np.log(umax[1]**2)
    ];

    return np.sum(Cu);

def Ccmp(u, inputs):
    dmax = inputs.CP_maxdistance;
    g = inputs.gravity_acc;
    m = inputs.joint_masses[0];

    utip = m*g*dmax;

    Ccmp = [
        #-np.log(utip**2 - u[0]**2) + np.log(utip**2),
        100*(u[0]/utip)**2,
        0
    ];

    return np.sum(Ccmp);

def cost(mpc_var, q, u, inputs):
    # MPC constants
    N  = mpc_var.q_num;
    Nu = mpc_var.u_num;
    P  = mpc_var.PH;
    u0 = inputs.prev_input;
    qd = [0, 0, 0, 0];

    # reshape input variable
    uc = np.reshape(u0 + u, [P+1, Nu]);

    # initialize cost array
    C = [0 for i in range(P+1)];

    for i in range(P+1):
        du = [uc[i][j] - uc[i-1][j] for j in range(Nu)];
        C[i] = C[i] + Cq(qd, q[i]);                 # state cost
        C[i-1] = C[i-1] + Cu(uc[i-1], du, inputs);  # input costs
        C[i-1] = C[i-1] + Ccmp(uc[i-1], inputs);    # CMP costs

    return np.sum(C);

class InputsALIP:
    def __init__(self, prev_input):
        self.gravity_acc          = -9.81;
        self.joint_masses         = [40];
        self.link_lengths         = [0.95];
        self.CP_maxdistance       = 0.1;
        self.input_bounds         = [100, 500];
        self.prev_input           = prev_input;

class Inputs3link:
    def __init__(self):
        self.num_inputs           = 3;
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0, 0, 0];
        self.joint_masses         = [5, 5, 30];
        self.link_lengths         = [0.5, 0.5, 0.6];

class IDVariables:
    def __init__(self, m1_inputs):
        self.model1   = m1_inputs;
        self.m1_mass  = MassMatrix_3link;
        self.m1_state = CoM_3link;
        self.m1_jacob = J_CoM_3link;

if __name__ == "__main__":
    # initialize inputs and math expressions class
    inputs_3link = Inputs3link();
    inputs_alip  = InputsALIP([0, 0]);
    id_3link     = IDVariables(inputs_3link);
    mathexp      = MathExpressions();

    # mpc variable parameters
    num_inputs  = 2;
    num_ssvar   = 2;
    PH_length   = 10;
    knot_length = 1;
    time_step   = 0.025;

    # desired state constants
    height = 0.95;
    theta  = math.pi/2;

    # MPC class variable
    mpc_alip = mpc.system('nno', cost, statespace_alip, inputs_alip, num_inputs, num_ssvar, PH_length, knot_length, time_step);
    mpc_alip.setMinTimeStep(1);

    # simulation variables
    sim_time = 0.6;  sim_dt = time_step;
    Nt = round(sim_time/time_step + 1);
    T = [i*sim_dt for i in range(Nt)];

    print(Nt);  print(T);

    # loop variables
    N_3link = inputs_3link.num_inputs;
    q_alip  = [[0 for i in range(num_ssvar)] for i in range(Nt)];
    q_desired = [[0 for i in range(2*num_ssvar)] for i in range(Nt)];
    q_3link = [[0 for i in range(2*N_3link)] for i in range(Nt)];
    u_alip  = [[0 for j in range(num_inputs*PH_length)] for i in range(Nt)];
    u_3link = [[0 for j in range(N_3link)] for i in range(Nt)];

    # initial state
    q_3link[0] = [0.357571103645510, 2.426450446298773, -1.213225223149386, 0, 0, 0];

    # used only for modeuler function
    sim_3link = mpc.system('nno', cost, statespace_3link, inputs_3link, N_3link, 2*N_3link, 1, 1, time_step);

    # simulation loop
    for i in range(1,Nt):
        print("\nt =", i*sim_dt);

        # convert state: 3link -> alip
        x_c = CoM_3link(q_3link[i-1], inputs_3link)[0];
        L = mathexp.base_momentum(q_3link[i-1][:N_3link], q_3link[i-1][N_3link:2*N_3link])[0][0];
        q_alip[i-1]  = [x_c, L];

        print("q_alip =\n", q_alip[i-1]);

        # solve the MPC problem w/ warmstarting
        inputs_alip = InputsALIP(u_alip[i-1][:num_inputs]);
        mpc_alip.setModelInputs(inputs_alip);
        (u_alip[i], C, n, brk, elapsed) = mpc_alip.solve(q_alip[i-1], u_alip[i-1]);

        print("u_alip =\n", u_alip[i][:2]);
        print("COST =", C);

        if (math.isnan(C)):
            break;

        # convert input: alip -> 3link
        q_desired[i] = [u_alip[i][0], height, theta, u_alip[i][1]];
        u_3link[i] = id.convert(id_3link, q_desired[i], q_3link[i], u_3link[i-1]);

        print("u_3link =\n", u_3link[i]);

        q_3link[i] = sim_3link.modeuler(q_3link[i-1], u_3link[i], 1)[1][-1];

        print("q_3link =\n", q_3link[i]);

    print(q_alip[0]);
    print(q_3link[0]);

    desiredStatePlot = plotStates_alip(T, [[q_desired[i][0], q_desired[i][3]] for i in range(Nt)]);
    statePlot = plotStates_alip(T, q_alip);
    plt.show();

    animation_3link(T, q_3link, inputs_3link);
