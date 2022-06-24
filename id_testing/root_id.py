import sys

sys.path.insert(0, 'models/.');

import mpc
from statespace_3link import *
from MathFunctionsCpp import MathExpressions
import inverse_dynamics as id
import math
import matplotlib.pyplot as plt


import gym
# from gym.wrappers import Monitor
import os
import numpy as np
import time
import argparse
import pickle
from datetime import datetime

from gym.envs.registration import registry, register, make, spec


class Inputs3link:
    def __init__(self):
        self.num_inputs           = 3;
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0, 0, 0];
        self.joint_masses         = [5, 5, 30];
        self.link_lengths         = [0.5, 0.5, 0.6];


if __name__ == "__main__":
	#==================== Create custom MuJoCo Environment ====================#
    render_mode = True;
    dynamics_randomization = False;
    apply_force = True;
    register(id='Pend3link-v0',
             entry_point='mujoco_envs.pend_3link:Pend3LinkEnv',
             kwargs =   {'dynamics_randomization': dynamics_randomization});
    env = gym.make('Pend3link-v0');
    state = env.reset();

    # initialize inputs and math expressions class
    mathexp = MathExpressions();
    inputs_3link = Inputs3link();

    # desired state constants
    height = 0.95;
    theta  = math.pi/2;

    # simulation variables
    sim_time = 10.0;  sim_dt = env.dt;	#env.dt = 0.0005
    Nt = round(sim_time/sim_dt + 1);
    T = [i*sim_dt for i in range(Nt)];

    # loop variables
    N_3link = inputs_3link.num_inputs;
    q_desired = [[0 for i in range(4)] for i in range(Nt)];
    q_3link = [[0 for i in range(2*N_3link)] for i in range(Nt+1)];
    u_3link = [[0 for j in range(N_3link)] for i in range(Nt)];
    ddq = [[0 for j in range(N_3link)] for i in range(Nt)];

    # Set initial state
    # init_state = np.array([0.357571103645510, 2.426450446298773, -1.213225223149386, 0.3, 0, 0]) + np.random.randn(6)*0.01
    init_state = np.array([0.7076, 1.7264, -0.8632, 0, 0, 0]) + np.random.randn(6)*0.01;
    q_3link[0] = init_state.tolist();

    u_previous = [0, 0, 0];
    u_alip_actual = [[0] for i in range(Nt)];

    env.set_state(init_state[0:3],init_state[3:6]);
    # simulation loop
    for i in range(Nt):
        # print("\nt =", i*sim_dt);

        # # calculate current CoM position and L for monitoring
        # (x_c, h_c, _) = CoM_3link(q_3link[i], inputs_3link);
        # # print("com: ",x_c)
        # L = mathexp.base_momentum(q_3link[i][:N_3link], q_3link[i][N_3link:2*N_3link])[0][0];

        q_desired[i] = [0, height, theta, 0];

        # convert input: alip -> 3link
        u_3link[i] = id.convert(inputs_3link, q_desired[i], q_3link[i], u_previous);

        if (u_3link[i] is None):
            # print("ERROR: ID-QP function returned None...");
            u_3link[i] = [0,0,0];
            break;

        action = np.array(u_3link[i])
        # print("action: ", action)

        next_state, _, _, _ = env.step(action);
        q_3link[i+1] = next_state.tolist();  # print(q_3link[i+1]);

        u_previous = u_3link[i];

        u_alip_actual[i] = [
            mathexp.centroidal_momentum(q_3link[i+1][:N_3link], q_3link[i+1][N_3link:2*N_3link])[0][0]
        ];
        print("Lc_actual:", u_alip_actual[i]);

        if render_mode:
            env.render();

    statePlot = plotStates_3link(T, q_3link[:Nt]);
    inputPlot = plotInputs_3link(T, u_3link);

    fig, amPlot = plt.subplots();
    amPlot.plot(T, u_alip_actual);

    plt.show(block=0);

    input("Press enter to close plots...");
    plt.close('all');
