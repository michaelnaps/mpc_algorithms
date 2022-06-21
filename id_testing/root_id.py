import sys

sys.path.insert(0, 'models/.');

import mpc
from statespace_3link import *
from modeuler import *
from MathFunctionsCpp import MathExpressions
import inverse_dynamics as id
import math
import matplotlib.pyplot as plt

class Inputs3link:
    def __init__(self):
        self.num_inputs           = 3;
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0, 0, 0];
        self.joint_masses         = [5, 5, 30];
        self.link_lengths         = [0.5, 0.5, 0.6];

if __name__ == "__main__":
    # initialize inputs and math expressions class
    mathexp = MathExpressions();
    inputs_3link = Inputs3link();

    # desired state constants
    height = 0.95;
    theta  = math.pi/2;

    # simulation variables
    sim_time = 2;  sim_dt = 0.001;
    Nt = round(sim_time/sim_dt + 1);
    T = [i*sim_dt for i in range(Nt)];

    # loop variables
    N_3link = inputs_3link.num_inputs;
    q_desired = [[0 for i in range(4)] for i in range(Nt)];
    q_3link = [[0 for i in range(2*N_3link)] for i in range(Nt)];
    u_3link = [[0 for j in range(N_3link)] for i in range(Nt)];

    # initial state
    # q_3link[0] = [math.pi/4, math.pi/4, math.pi/4, 0, 0, 0];
    # q_3link[0] = [math.pi/3, math.pi/3, math.pi/3, 0, 0, 0];
    q_3link[0] = [0.357571103645510, 2.426450446298773, -1.213225223149386, 0.3, 0, 0];

    # simulation loop
    for i in range(Nt-1):
        print("\nt =", i*sim_dt);

        # calculate current CoM position and L for monitoring
        (x_c, h_c, _) = CoM_3link(q_3link[i], inputs_3link);
        L = mathexp.base_momentum(q_3link[i][:N_3link], q_3link[i][N_3link:2*N_3link])[0][0];

        q_desired[i] = [0, height, theta, 0];

        # convert input: alip -> 3link
        u_3link[i] = id.convert(inputs_3link, q_desired[i], q_3link[i], 1);
        print(u_3link[i]);

        if (u_3link[i] is None):
            print("ERROR: ID-QP function returned None...");
            break;

        q_3link[i+1] = modeuler(statespace_3link, sim_dt, sim_dt, q_3link[i], u_3link[i], inputs_3link)[1][-1];

        print("modeuler:", q_3link[i+1]);

    ans = input("\nSee animation? [y/n] ");
    if (ans == 'y'):  animation_3link(T[:i+2], q_3link[:i+2], inputs_3link);
