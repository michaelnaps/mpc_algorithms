import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import mpc
from statespace_alip import *
import matplotlib.pyplot as plt
import numpy.random as rd

def Cq(qd, q):
    kx = 250;  kL = 10;
    return (kx*(qd[0] - q[0])**2 + kL*(qd[1] - q[1])**2);

def Cu(u, du, inputs):
    kdu = 100;  ku = 1;
    umax = inputs.input_bounds;
    u_error = umax[0]**2 - u[0]**2;
    return (kdu*(du[0])**2 - ku*np.log(u_error) + ku*np.log(umax[0]**2));

def Ccmp(q, inputs):
    kcmp = 100;
    dmax = inputs.CP_maxdistance;
    g = inputs.gravity_acc;
    m = inputs.joint_masses[0];
    utip = m*g*dmax;
    return kcmp*(q[1]/utip)**2;

def cost(mpc_var, q, u, inputs):
    # MPC constants
    N  = mpc_var.q_num;
    Nu = mpc_var.u_num;
    P  = mpc_var.PH;
    u0 = inputs.prev_input;
    qd = [0, 0];

    # reshape input variable
    uc = np.reshape(u0 + u, [P+1, Nu]);

    # initialize cost array
    C = [0 for i in range(P+1)];

    for i in range(P+1):
        du = [uc[i][j] - uc[i-1][j] for j in range(Nu)];
        C[i] = C[i] + Cq(qd, q[i]);                 # state cost
        C[i-1] = C[i-1] + Cu(uc[i-1], du, inputs);  # input costs
        C[i] = C[i] + Ccmp(q[i], inputs);    # CMP costs

    return np.sum(C);

class InputVariables:
    def __init__(self, prev_input):
        self.gravity_acc          = -9.81;
        self.joint_masses         = [80];
        self.link_lengths         = [2.0];
        self.CP_maxdistance       = 0.1;
        self.input_bounds         = [50];
        self.prev_input           = prev_input;

def updateFunction(mpc_var, q, u):
    N = mpc_var.u_num;
    return InputVariables(u[:N]);

def main():
    inputs = InputVariables([0]);

    # mpc variables
    num_inputs  = 1;
    num_ssvar   = 2;
    PH_length   = 10;
    knot_length = 2;
    time_step = 0.05;

    disturb = 0.05*rd.random();
    q0 = [0-disturb, 0];
    u0 = [0 for i in range(num_inputs*PH_length)];

    # initialize mpc system
    mpc_var = mpc.system('nno', cost, statespace_alip, inputs, num_inputs, num_ssvar, PH_length, knot_length, time_step);
    # mpc_var.setAlpha(25);
    # mpc_var.setAlphaMethod('bkl')
    mpc_var.setMinTimeStep(1);  # very large (no adjustment)

    # simulate results
    sim_results = mpc_var.sim_root(10, q0, u0, updateFunction, output=1);

    # save results
    saveResults_alip("results.pickle", sim_results);

    # report results
    reportResults_alip(sim_results, inputs);

if __name__ == "__main__":
    main();
