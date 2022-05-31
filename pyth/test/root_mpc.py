import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import mpc
from statespace_alip import *
import matplotlib.pyplot as plt

def Cq(qd, q):
    Cq = [
        100*(qd[0] - q[0])**2 +  1*(qd[2] - q[2])**2,
         10*(qd[1] - q[1])**2 +    (qd[3] - q[3])**2
    ];

    return np.sum(Cq);

def Cu(u, du, inputs):
    umax = inputs.input_bounds;

    Cu = [
        1e-5*(du[0])**2 - np.log(umax[0]**2 - u[0]**2) + np.log(umax[0]**2),
        1e-5*(du[1])**2# - np.log(umax[1]**2 - u[1]**2) + np.log(umax[1]**2)
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
    uc = np.reshape(u, [P, Nu]);

    C = [0 for i in range(P+1)];

    # calculate change in input
#    du = [[0 for j in range(Nu)] for i in range(P)];
#    du[0] = [uc[0][i] - u0[i] for i in range(Nu)];
#    for i in range(1, P):
#        for j in range(Nu):
#            du[i][j] = uc[i][j] - uc[i-1][j];

    for i in range(P+1):
        C[i] = C[i] + Cq(qd, q[i]);  # state cost
        if i != P:
#            C[i] = C[i] + Cu(uc[i], du[i], inputs);  # input costs
            C[i] = C[i] + Ccmp(uc[i], inputs);  # CMP costs

    return np.sum(C);

class InputVariables:
    def __init__(self, prev_input):
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0];
        self.joint_masses         = [80];
        self.link_lengths         = [2.0];
        self.CP_maxdistance       = 0.1;
        self.input_bounds         = [60, 150];
        self.prev_input           = prev_input;

def main():
    inputs = InputVariables([0, 0]);

    # mpc variables
    num_inputs = 2;
    num_ssvar = 2;
    PH_length = 4;
    knot_length = 2;

    q0 = [0-0.01, 0, 0, 0];
    u0 = [0 for i in range(num_inputs*PH_length)];

    mpc_var = mpc.system('nno', cost, inputs, num_inputs, num_ssvar, statespace_alip, PH_length, knot_length);

    sim_results = mpc_var.sim_root(2, q0, u0, 1);

    reportResults_alip(sim_results, inputs);

if __name__ == "__main__":
    main();
