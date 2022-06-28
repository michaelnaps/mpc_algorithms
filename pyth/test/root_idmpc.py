import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import mpc
from statespace_alip import *
from statespace_tpm import *
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
    kx = 2500;  kL = 10;
    return (kx*(qd[0] - q[0])**2 + kL*(qd[1] - q[1])**2);

def Cu(u, du, inputs):
    kdu = 100;  ku = 1;
    umax = inputs.input_bounds;
    u_error = umax[0]**2 - u[0]**2;
    return (100*(du[0])**2 - ku*np.log(u_error) + ku*np.log(umax[0]**2));

def Ccmp(u, inputs):
    kcmp = 100;
    dmax = inputs.CP_maxdistance;
    g = inputs.gravity_acc;
    m = inputs.joint_masses[0];
    utip = m*g*dmax;
    return kcmp*(u[0]/utip)**2;

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
        self.CP_maxdistance       = 0.1;
        self.input_bounds         = [100];
        self.prev_input           = prev_input;

class InputsTPM:
    def __init__(self):
        self.num_inputs           = 3;
        self.gravity_acc          = -9.81;
        self.damping_coefficients = [0, 0, 0];
        self.joint_masses         = [5, 5, 30];
        self.link_lengths         = [0.5, 0.5, 0.6];

if __name__ == "__main__":
    #==== Create custom MuJoCo Environment ====#
    dynamics_randomization = 0;
    apply_force = 1;
    register(id='Pend3link-v0',
            entry_point='mujoco_envs.pend_3link:Pend3LinkEnv',
            kwargs={'dynamics_randomization': dynamics_randomization});
    env = gym.make('Pend3link-v0');
    state = env.reset();

    # initialize inputs and math expressions class
    inputs_tpm = InputsTPM();
    inputs_alip  = InputsALIP([0]);
    mathexp      = MathExpressions();

    # mpc variable parameters
    num_inputs  = 1;
    num_ssvar   = 2;
    PH_length   = 20;
    knot_length = 1;
    time_step   = 0.05;

    # desired state constants
    height = 0.95;
    theta  = np.pi/2;

    # MPC class variable
    mpc_alip = mpc.system('nno', cost, statespace_alip, inputs_alip, num_inputs,
                          num_ssvar, PH_length, knot_length, time_step, max_iter=10);
    # mpc_alip.setAlpha(25);
    # mpc_alip.setAlphaMethod('bkl');
    mpc_alip.setMinTimeStep(1);

    # simulation variables
    sim_time = 60.0;  sim_dt = env.dt;
    Nt = round(sim_time/sim_dt + 1);
    T = [i*sim_dt for i in range(Nt)];

    # loop variables
    N_tpm = inputs_tpm.num_inputs;
    q_alip  = [[0 for i in range(num_ssvar)] for i in range(Nt)];
    q_desired = [[0 for i in range(4)] for i in range(Nt)];
    q_tpm = [[0 for i in range(2*N_tpm)] for i in range(Nt)];
    u_alip  = [[0 for j in range(num_inputs*PH_length)] for i in range(Nt)];
    u_tpm = [[0 for j in range(N_tpm)] for i in range(Nt)];

    s_actual = [[0, 0, 0, 0] for i in range(Nt)];

    # MPC variables
    Clist = [0 for i in range(Nt)];
    nlist = [0 for i in range(Nt)];
    brklist = [100 for i in range(Nt)];
    tlist = [0 for i in range(Nt)];

	# SET INITIAL STATES (TPM and ALIP)
    init_state = np.array([1.0236756190034337, 1.1651000155300129, -0.6137993852195395, 0, 0, 0]);
    init_state = init_state + 0.1*np.random.randn(6);
    q_tpm[0] = init_state.tolist();
    env.set_state(init_state[0:3],init_state[3:6]);

    (x_c, h_c, q_c) = CoM_tpm(q_tpm[0], inputs_tpm);
    L   = mathexp.base_momentum(q_tpm[0][:N_tpm], q_tpm[0][N_tpm:2*N_tpm])[0][0];
    L_c = mathexp.centroidal_momentum(q_tpm[0][:N_tpm], q_tpm[0][N_tpm:2*N_tpm])[0][0];

    q_alip[0] = [x_c, L];
    s_actual[0] = [x_c, h_c, q_c, L_c];

    print("INITIAL STATE:");
    print("TPM:", q_tpm[0]);
    print("ALIP:", q_alip[0]);
    print("ID-QP:", s_actual[0]);

    # simulation loop
    for i in range(1,Nt):
        print("\nt =", i*sim_dt);

        print("desired alip state:", q_desired[i-1]);
        print("current alip state:", s_actual[i-1]);

        # set alip model inputs
        inputs_alip.prev_inputs = u_alip[i-1][:num_inputs];
        inputs_alip.link_lengths = [h_c];
        mpc_alip.setModelInputs(inputs_alip);

        # solve MPC problem
        if ((i-1) % 25) == 0:
            (u_alip[i], Clist[i], nlist[i], brklist[i], tlist[i]) = mpc_alip.solve(q_alip[i-1], u_alip[i-1], output=0);
            x_d = mpc_alip.simulate(u_alip[i-1], u_alip[i])[1][0];
        else:
            u_alip[i] = u_alip[i-1];
            Clist[i]  = Clist[i-1];
            nlist[i]  = -1;
            brklist[i] = 100;
            tlist[i]  = -1;

        # construct q_d for ID-QP function
        q_desired[i] = [x_d, height, theta, u_alip[i][0]];

        if (np.isnan(Clist[i])):
            print("ERROR: Cost is nan after optimization...");
            break;

        # convert input: alip -> tpm
        u_tpm[i] = id.convert(inputs_tpm, q_desired[i], q_tpm[i-1], u_tpm[i-1]);

        if (u_tpm[i] is None):
            print("ERROR: ID-QP function returned None...");
            u_tpm[i] = [0,0,0];
            break;

        # TPM simulation step
        q_tpm[i] = env.step(u_tpm[i])[0].tolist();

        # convert state: tpm -> alip
        (x_c, h_c, q_c) = CoM_tpm(q_tpm[i], inputs_tpm);
        L   = mathexp.base_momentum(q_tpm[i][:N_tpm], q_tpm[i][N_tpm:2*N_tpm])[0][0];
        L_c = mathexp.centroidal_momentum(q_tpm[i][:N_tpm], q_tpm[i][N_tpm:2*N_tpm])[0][0];

        q_alip[i] = [x_c, L];
        s_actual[i] = [x_c, h_c, q_c, L_c];

    ans = input("\nSee animation? [y/n] ");
    if (ans == 'y'):
        for i in range(Nt):
            env.set_state(np.array(q_tpm[i][0:3]), np.array(q_tpm[i][3:6]));
            env.render();
            time.sleep(0.0005);

        input("Press enter to exit program...");

    alip_results = (T, q_alip, u_alip, Clist, nlist, brklist, tlist);
    tpm_results  = (T, q_tpm[:Nt], u_tpm, s_actual);

    saveResults_alip("resultsALIP.pickle", alip_results);
    saveResults_tpm("resultsTPM.pickle", tpm_results);
