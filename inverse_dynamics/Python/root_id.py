import sys

sys.path.insert(0, 'models/.');

from statespace_alip import *
from statespace_3link import *
from modeuler import *

import inverse_dynamics as id
import matplotlib.pyplot as plt
import math

class MPCVariables:
    def __init__(self):
        self.sim_time     = 3;
        self.model        = statespace_3link;
        # self.cost_state   = Cq;
        # self.cost_input   = Cu;
        # self.cost_CMP     = Ccmp;
        self.PH_length    = 1;
        self.knot_length  = 1;
        self.time_step    = 0.025;
        self.appx_zero    = 1e-6;
        self.step_size    = 1e-3;
        self.num_ssvar    = 3;
        self.num_inputs   = 3;
        self.des_config   = [0, 0, 0, 0];
        self.max_iter     = 5;

class IDVariables:
    def __init__(self, m1_inputs):
        self.model1   = m1_inputs;
        self.m1_dyn   = statespace_3link;
        self.m1_Jdim  = 3;
        self.m1_state = CoM_3link;
        self.m1_jacob = J_CoM_3link;

        self.m2_desired = [0, 0.5, math.pi/2];
        # self.m2_inputs  = inputs_alip;

class InputVariables:
    def __init__(self):
        self.num_inputs           = 3;
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0, 0, 0];
        self.joint_masses         = [5, 5, 30];
        self.link_lengths         = [0.4, 0.4, 0.6];
        self.input_bounds         = [1, 1, 1];

def main():
    inputs_3link = InputVariables();
    id_var = IDVariables(inputs_3link);
    mpc_var = MPCVariables();

    # simulation time variables
    sim_time = 3;  dt = 0.025;
    Nt = int(sim_time/dt) + 1;
    T = [i*dt for i in range(Nt)];

    # state variables
    u = [[0 for j in range(inputs_3link.num_inputs)] for i in range(Nt)];
    q = [[0 for j  in range(2*inputs_3link.num_inputs)] for i in range(Nt)];
    q[0] = [math.pi/4, math.pi/2, -math.pi/4, 0, 0, 0];

    for i in range(1,Nt):
        u[i] = id.convert(id_var, q[i-1]);
        q[i] = modeuler(mpc_var, q[i-1], u[i], inputs_3link)[1][-1];

    ans = input("\nShow static plots? [y/n] ");
    if ans == 'y':
        statePlot = plotStates_3link(T, q);
        inputPlot = plotInputs_3link(T, u);
        plt.show();

    ans = input("\nShow animation? [y/n] ");
    if ans == 'y':
        animation_3link(T, q, inputs_3link);

if __name__ == "__main__":
    main();
