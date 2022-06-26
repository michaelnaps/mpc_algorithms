import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import mpc
from statespace_alip import *
from statespace_3link import *
from MathFunctionsCpp import MathExpressions
import inverse_dynamics as id
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

def Cq(qd, q):
    Cq = [
        100*(qd[0] - q[0])**2,
         10*(qd[1] - q[1])**2
    ];

    return np.sum(Cq);

def Cu(u, du, inputs):
    umax = inputs.input_bounds;

    u_error = [
        umax[0]**2 - u[0]**2
    ];

    Cu = [
        100*(du[0])**2 - np.log(u_error[0]) + np.log(umax[0]**2)
    ];

    return np.sum(Cu);

def Ccmp(u, inputs):
    dmax = inputs.CP_maxdistance;
    g = inputs.gravity_acc;
    m = inputs.joint_masses[0];

    utip = m*g*dmax;

    Ccmp = [
        100*(u[0]/utip)**2
    ];

    return np.sum(Ccmp);

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
        C[i-1] = C[i-1] + Ccmp(uc[i-1], inputs);    # CMP costs

    return np.sum(C);

class InputsALIP:
    def __init__(self, prev_input):
        self.gravity_acc          = -9.81;
        self.joint_masses         = [40];
        self.link_lengths         = [0.95];
        self.CP_maxdistance       = 0.5;
        self.input_bounds         = [2];
        self.prev_input           = prev_input;

class Inputs3link:
    def __init__(self):
        self.num_inputs           = 3;
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0, 0, 0];
        self.joint_masses         = [5, 5, 30];
        self.link_lengths         = [0.5, 0.5, 0.6];

if __name__ == "__main__":
    #==== Create custom MuJoCo Environment ====#
    render_mode = 0;
    dynamics_randomization = 0;
    apply_force = 1;
    register(id='Pend3link-v0',
            entry_point='mujoco_envs.pend_3link:Pend3LinkEnv',
            kwargs={'dynamics_randomization': dynamics_randomization});
    env = gym.make('Pend3link-v0');
    state = env.reset();

    # initialize inputs and math expressions class
    inputs_3link = Inputs3link();
    inputs_alip  = InputsALIP([0]);
    mathexp      = MathExpressions();

    # mpc variable parameters
    num_inputs  = 1;
    num_ssvar   = 2;
    PH_length   = 10;
    knot_length = 1;
    time_step   = 0.025;

    # desired state constants
    height = 0.95;
    theta  = np.pi/2;

    # MPC class variable
    mpc_alip = mpc.system('ngd', cost, statespace_alip, inputs_alip, num_inputs,
                          num_ssvar, PH_length, knot_length, time_step, max_iter=100);
    mpc_alip.setAlpha(25);
    mpc_alip.setAlphaMethod('bkl');
    mpc_alip.setMinTimeStep(1);

    # simulation variables
    sim_time = 5.0;  sim_dt = env.dt;
    Nt = round(sim_time/sim_dt + 1);
    T = [i*sim_dt for i in range(Nt)];

    # loop variables
    N_3link = inputs_3link.num_inputs;
    q_alip  = [[0 for i in range(num_ssvar)] for i in range(Nt)];
    q_desired = [[0 for i in range(4)] for i in range(Nt)];
    q_3link = [[0 for i in range(2*N_3link)] for i in range(Nt)];
    u_alip  = [[0 for j in range(num_inputs*PH_length)] for i in range(Nt)];
    u_3link = [[0 for j in range(N_3link)] for i in range(Nt)];

    u_alip_actual = [[0] for i in range(Nt)];

    # MPC variables
    Clist = [0 for i in range(Nt)];
    nlist = [0 for i in range(Nt)];
    brklist = [0 for i in range(Nt)];
    tlist = [0 for i in range(Nt)];

	# Set initial state
    init_state = np.array([1.0236756190034337, 1.1651000155300129, -0.6137993852195395, 0, 0, 0]);
    #init_state = init_state + 0.01*np.random.randn(6);
    q_3link[0] = init_state.tolist();
    env.set_state(init_state[0:3],init_state[3:6]);

    # simulation loop
    for i in range(1,Nt):
        print("\nt =", i*sim_dt);
        print("current state:", q_3link[i-1]);

        # convert state: 3link -> alip
        (x_c, h_c, _) = CoM_3link(q_3link[i-1], inputs_3link);
        L = mathexp.base_momentum(q_3link[i-1][:N_3link], q_3link[i-1][N_3link:2*N_3link])[0][0];
        q_alip[i-1] = [x_c, L];

        # set alip model inputs
        inputs_alip.prev_inputs = u_alip[i-1][:num_inputs];
        # inputs_alip.link_lengths = [h_c];
        mpc_alip.setModelInputs(inputs_alip);

        # solve MPC problem
        if ((i-1) % 10) == 0:
            (u_alip[i], Clist[i], nlist[i], brklist[i], tlist[i]) = mpc_alip.solve(q_alip[i-1], u_alip[i-1], output=0);
        else:
            u_alip[i] = u_alip[i-1];
            Clist[i]  = Clist[i-1];
            nlist[i]  = 0;
            brklist[i] = 3;
            tlist[i]  = 0;

        q_temp = mpc_alip.simulate(q_alip[i-1], u_alip[i]);
        x_desired = q_temp[1][0];  L_desired = q_temp[1][1];

        if (np.isnan(Clist[i])):
            print("ERROR: Cost is nan after optimization...");
            break;
        else:
            print("cost post-optimization:", Clist[i]);

        print("mpc results:", u_alip[i])

        # q_desired[i] = [0, height, theta, 0];
        q_desired[i] = [0, height, theta, u_alip[i][0]];
        print("desired conversion variables:", q_desired[i]);

        # convert input: alip -> 3link
        u_3link[i] = id.convert(inputs_3link, q_desired[i], q_3link[i-1], u_3link[i-1]);

        if (u_3link[i] is None):
            print("ERROR: ID-QP function returned None...");
            u_3link[i] = [0,0,0];
            break;

        action = np.array(u_3link[i]);
        print("action:", u_3link[i]);

        next_state, _, _, _ = env.step(action);
        q_3link[i] = next_state.tolist();

        u_alip_actual[i] = [
            mathexp.centroidal_momentum(q_3link[i][:N_3link], q_3link[i][N_3link:2*N_3link])[0][0]
        ];
        print("Lc_actual:", u_alip_actual[i]);

        if render_mode:
            env.render();

    # ans = input("\nSee animation? [y/n] ");
    # if (ans == 'y'):  animation_3link(T, q_3link, inputs_3link);

    alip_results = (T, q_alip, u_alip, Clist, nlist, brklist, tlist);
    tpm_results  = (T, q_3link[:Nt], u_3link);

    reportResults_alip(alip_results, inputs_alip);


    fig, inputComparison = plt.subplots();
    inputComparison.plot(T, np.transpose(u_alip)[0], label="desired");
    inputComparison.plot(T, u_alip_actual, label="actual");
    plt.legend();

    statePlot_3link = plotStates_3link(T, q_3link[:Nt]);
    inputPlot_3link = plotInputs_3link(T, u_3link);
    plt.show(block=0);

    input("Press enter to close plots...");
    plt.close('all');
