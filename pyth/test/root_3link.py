import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import mpc
from statespace_3link import *
import matplotlib.pyplot as plt
import numpy.random as rd

def Cq(qd, q):
    Cq = [
        100*(qd[0] - q[0])**2 + (qd[3] - q[3])**2,
        100*(qd[1] - q[1])**2 + (qd[4] - q[4])**2,
        100*(qd[2] - q[2])**2 + (qd[5] - q[5])**2
    ];

    return np.sum(Cq);

def Cu(u, du, inputs):
    umax = inputs.input_bounds;

    u_error = [
        umax[0]**2 - u[0]**2,
        umax[1]**2 - u[1]**2,
        umax[2]**2 - u[2]**2
    ];

    Cu = [
        1e-5*(du[0])**2 - np.log(u_error[0]) + np.log(umax[0]**2),
        1e-5*(du[1])**2 - np.log(u_error[1]) + np.log(umax[1]**2),
        1e-5*(du[2])**2 - np.log(u_error[2]) + np.log(umax[2]**2),
    ];

    return np.sum(Cu);

def Ccmp(u, inputs):
    dmax = inputs.CP_maxdistance;
    g = inputs.gravity_acc;
    m = inputs.joint_masses[0];

    utip = m*g*dmax;

    Ccmp = [
        #-np.log(utip**2 - u[0]**2) + np.log(utip**2),
        100*(u[0]/utip)**2
        # 0
    ];

    return np.sum(Ccmp);

def cost(mpc_var, q, u, inputs):
    # MPC constants
    N  = mpc_var.q_num;
    Nu = mpc_var.u_num;
    P  = mpc_var.PH;
    u0 = inputs.prev_input;
    qd = [np.pi/4, np.pi/2, -np.pi/4, 0, 0, 0];

    # reshape input variable
    uc = np.reshape(u0 + u, [P+1, Nu]);

    # initialize cost array
    C = [0 for i in range(P+1)];

    for i in range(P+1):
        du = [uc[i][j] - uc[i-1][j] for j in range(Nu)];
        C[i] = C[i] + Cq(qd, q[i]);                 # state cost
        C[i-1] = C[i-1] + Cu(uc[i-1], du, inputs);  # input costs
        # C[i-1] = C[i-1] + Ccmp(uc[i-1], inputs);    # CMP costs

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
    PH_length   = 10;
    knot_length = 1;

    q0 = [np.pi/2, 0, 0, 0, 0, 0];
    u0 = [0 for i in range(num_inputs*PH_length)];

    mpc_var = mpc.system('ngd', cost, statespace_3link, inputs, num_inputs,
                         num_ssvar, PH_length, knot_length,
                         time_step=0.025, appx_zero=1e-6, step_size=1e-3, max_iter=10);
    mpc_var.setAlpha(50);
    mpc_var.setAlphaMethod('bkl');
    mpc_var.setMinTimeStep(1);  # very large (no adjustment)

    sim_results = mpc_var.sim_root(1, q0, u0, updateFunction, 1);

    animation_3link(sim_results[0], sim_results[1], inputs);
