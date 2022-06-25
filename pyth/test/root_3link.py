import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import mpc
from statespace_3link import *
from MathFunctionsCpp import MathExpressions
import matplotlib.pyplot as plt
import numpy.random as rd

def Cq(qd, q):
    kd = 1;

    Cq = [
        100*(qd[0] - q[0])**2 + kd*(qd[4] - q[4])**2,
        150*(qd[1] - q[1])**2 + kd*(qd[5] - q[5])**2,
         50*(qd[2] - q[2])**2 + kd*(qd[6] - q[6])**2,
        200*(qd[3] - q[3])**2 + kd*(qd[7] - q[7])**2
    ];

    return np.sum(Cq);

def Cu(u, du, inputs):
    umax = inputs.input_bounds;
    ku = 1e-5;

    u_error = [
        umax[0]**2 - u[0]**2,
        umax[1]**2 - u[1]**2,
        umax[2]**2 - u[2]**2
    ];

    Cu = [
        ku*(du[0])**2 - ku*np.log(u_error[0]) + ku*np.log(umax[0]**2),
        ku*(du[1])**2 - ku*np.log(u_error[1]) + ku*np.log(umax[1]**2),
        ku*(du[2])**2 - ku*np.log(u_error[2]) + ku*np.log(umax[2]**2),
    ];

    return np.sum(Cu);

def cost(mpc_var, q, u, inputs):
    mathexp = MathExpressions();

    # MPC constants
    N  = mpc_var.q_num;
    Nq = int(N/2);
    Nu = mpc_var.u_num;
    P  = mpc_var.PH;

    # cost variables
    u0 = inputs.prev_input;
    qd = [0, 0.95, np.pi/2, 0, 0, 0, 0, 0];

    # reshape input variable
    uc = np.reshape(u0 + u, [P+1, Nu]);

    # initialize cost array
    C = [0 for i in range(P+1)];

    for i in range(P+1):
        # calculate dynamics variables
        (x_c, h_c, q_l3) = CoM_3link(q[i], inputs);
        (J_a, dJ_a) = J_CoM_3link(q[i], inputs);

        L_c = mathexp.centroidal_momentum(q[i][:Nq], q[i][Nq:N])[0][0];
        JL_c = mathexp.J_centroidal_momentum(q[i][:Nq]);

        dq_c = np.matmul(J_a, q[i][Nq:N]);

        q_c  = [x_c, h_c, q_l3, 0, dq_c[0], dq_c[1], dq_c[2], 0];

        # change in input cost
        du = [0, 0, 0];
        if i > 0:  du = [uc[i][j] - uc[i-1][j] for j in range(Nu)];

        # window cost
        C[i] = C[i] + Cq(qd, q_c) + Cu(u, du, inputs);

    return np.sum(C);

class Inputs3link:
    def __init__(self, prev_input):
        self.gravity_acc          = -9.81;
        self.num_inputs           = 3;
        self.damping_coefficients = [0, 0, 0];
        self.joint_masses         = [5, 5, 30];
        self.link_lengths         = [0.5, 0.5, 0.6];
        self.input_bounds         = [1000, 1000, 1000];
        self.prev_input           = prev_input;

def updateFunction(mpc_var, q, u):
    N = mpc_var.u_num;
    return Inputs3link(u[:N]);

if (__name__ == "__main__"):
    inputs = Inputs3link([0, 0, 0]);

    # mpc variables
    num_inputs  = 3;
    num_ssvar   = 6;
    PH_length   = 1;
    knot_length = 1;

    q0 = [np.pi/4, np.pi/2, -np.pi/4, 0, 0, 0];
    u0 = [0 for i in range(num_inputs*PH_length)];

    mpc_var = mpc.system('nno', cost, statespace_3link, inputs, num_inputs,
                         num_ssvar, PH_length, knot_length,
                         time_step=0.0005, appx_zero=1e-6, step_size=1e-3, max_iter=100);
    # mpc_var.setAlpha(50);
    # mpc_var.setAlphaMethod('bkl');
    # mpc_var.setMinTimeStep(1);  # very large (no adjustment)

    sim_results = mpc_var.sim_root(5, q0, u0, updateFunction, 1);

    statePlot = plotStates_3link(sim_results[0], sim_results[1]);
    inputPlot = plotInputs_3link(sim_results[0], sim_results[2]);
    costPlot  = plotCost_3link(sim_results[0], sim_results[3]);
    brkPlot   = plotBrkFreq_3link(sim_results[5]);
    plt.show(block=0);

    input("Press enter to close plots...");
    plt.close('all');

    # animation_3link(sim_results[0], sim_results[1], inputs);
