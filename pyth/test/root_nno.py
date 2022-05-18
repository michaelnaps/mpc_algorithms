import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import nno
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

class MPCVariables:
    def __init__(self):
        self.sim_time     = 3;
        self.model        = statespace_alip;
        self.cost_state   = Cq;
        self.cost_input   = Cu;
        self.cost_CMP     = Ccmp;
        self.PH_length    = 4;
        self.knot_length  = 2;
        self.time_step    = 0.025;
        self.appx_zero    = 1e-6;
        self.step_size    = 1e-3;
        self.num_ssvar    = 2;
        self.num_inputs   = 2;
        self.des_config   = [0, 0, 0, 0];
        self.max_iter     = 5;

class InputVariables:
    def __init__(self):
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0];
        self.joint_masses         = [80];
        self.link_lengths         = [2.0];
        self.CP_maxdistance       = 0.1;
        self.input_bounds         = [60, 150];

def main():
    inputs  = InputVariables();
    mpc_var = MPCVariables();

    q0 = [0, 0, 0, 0];
    u0 = [0 for i in range(mpc_var.num_inputs*mpc_var.PH_length)];
    u_d = [0.5, 1.5, [1000, 0]];

    mpc_results = nno.mpc_root(mpc_var, q0, u0, u_d, inputs, 1);

    reportResults_alip(inputs, mpc_var, mpc_results);

if __name__ == "__main__":
    main();
