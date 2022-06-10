import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import mpc
from statespace_alip import *
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
        1e-5*(du[1])**2 # - np.log(umax[1]**2 - u[1]**2) + np.log(umax[1]**2)
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

class InputVariables:
    def __init__(self, prev_input):
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0];
        self.joint_masses         = [80];
        self.link_lengths         = [2.0];
        self.CP_maxdistance       = 0.1;
        self.input_bounds         = [60, 150];
        self.prev_input           = prev_input;

def updateFunction(mpc_var, q, u):
    N = mpc_var.u_num;
    return InputVariables(u[:N]);

def main():
    inputs = InputVariables([0, 0]);

    # mpc variables
    num_inputs  = 2;
    num_ssvar   = 2;
    PH_length   = 5;
    knot_length = 2;

    q0 = [0-0.05, 0];
    u0 = [0 for i in range(num_inputs*PH_length)];

    mpc_var = mpc.system('nno', cost, statespace_alip, inputs, num_inputs, num_ssvar, PH_length, knot_length);
    mpc_var.setMinTimeStep(1);

    sim_results = mpc_var.sim_root(2, q0, u0, updateFunction, 1);

    reportResults_alip(sim_results, inputs);
    saveResults_alip("prevRun_data.pickle", sim_results);

if __name__ == "__main__":
    main();
