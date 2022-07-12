import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth');
sys.path.insert(0, 'models');

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
    return (kdu*(du[0])**2 - ku*np.log(u_error) + ku*np.log(umax[0]**2));

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
        self.link_lengths         = [1.0];
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

# IMPORTANT VARIABLE NOTATION:
#   q - TPM state variables
#   u - TPM input variables
#   s - ALIP state variables
#   v - ALIP input variables
#   z - ALIP:TPM conversion attributes

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
    PH_length   = 10;
    knot_length = 1;
    time_step   = 0.05;

    # desired state constants
    height = 1.0;
    theta  = np.pi/2;

    # mid-sim change in height
    dh = [0.0, 1.0];
    dh_next = [0.0, 1.0];

    # apply disturbance
    t_disturb = [0.0, 0.0, 0.0];

    # MPC class variable
    mpc_alip = mpc.system('nno', cost, statespace_alip, inputs_alip, num_inputs,
                          num_ssvar, PH_length, knot_length, time_step, max_iter=10);
    # mpc_alip.setAlpha(25);
    # mpc_alip.setAlphaMethod('bkl');
    mpc_alip.setMinTimeStep(1);

    # simulation variables
    sim_time = 30.0;  sim_dt = env.dt;
    Nt = round(sim_time/sim_dt + 1);
    T = [i*sim_dt for i in range(Nt)];

    # loop variables
    N_tpm = inputs_tpm.num_inputs;
    s  = [[0 for i in range(num_ssvar)] for i in range(Nt)];
    z_d = [[0 for i in range(4)] for i in range(Nt)];
    q = [[0 for i in range(2*N_tpm)] for i in range(Nt)];
    v  = [[0 for j in range(num_inputs*PH_length)] for i in range(Nt)];
    u = [[0 for j in range(N_tpm)] for i in range(Nt)];

    z_a = [[0, 0, 0, 0] for i in range(Nt)];

    # MPC variables
    Clist = [0 for i in range(Nt)];
    nlist = [0 for i in range(Nt)];
    brklist = [100 for i in range(Nt)];
    tlist = [0 for i in range(Nt)];

	# SET INITIAL STATES (TPM and ALIP)
    init_state = np.array([1.0236756190034337, 1.1651000155300129, -0.6137993852195395, 0, 0, 0]);
    # init_state = np.array([0.8217631258316602, 1.1976983872266735, -0.5667525458100534, -0.09402431958351876, 0.16078048669562325, -0.10869250496817745]);
    init_state = init_state + 0.1*np.random.randn(6);
    q[0] = init_state.tolist();
    env.set_state(init_state[0:3],init_state[3:6]);

    (x_c, h_c, q_c) = CoM_tpm(q[0], inputs_tpm);
    L   = mathexp.base_momentum(q[0][:N_tpm], q[0][N_tpm:2*N_tpm])[0][0];
    L_c = mathexp.centroidal_momentum(q[0][:N_tpm], q[0][N_tpm:2*N_tpm])[0][0];

    s[0] = [x_c, L];
    z_a[0] = [x_c, h_c, q_c, L_c];

    print("INITIAL STATE:");
    print("TPM:", q[0]);
    print("ALIP:", s[0]);
    print("ID-QP:", z_a[0]);

    # simulation loop
    for i in range(1,Nt):
        print("\nt =", i*sim_dt);

        print("desired alip attributes: [%.6f, %.6f, %.6f, %.12f]"
              % (z_d[i-1][0], z_d[i-1][1], z_d[i-1][2], z_d[i-1][3]));
        print("current alip attrubutes: [%.6f, %.6f, %.6f, %.12f]"
              % (z_a[i-1][0], z_a[i-1][1], z_a[i-1][2], z_a[i-1][3]));

        # check for a change in height
        if (np.abs(i*sim_dt - dh[0]) < 1e-6):
            height = dh[1];

            if dh_next[0] != -1:
                dh = dh_next;

        # set alip model inputs
        inputs_alip.prev_inputs = v[i-1][:num_inputs];
        inputs_alip.link_lengths = [h_c];
        mpc_alip.setModelInputs(inputs_alip);

        # solve MPC problem
        if (((i-1) % 25) == 0):
            (v[i], Clist[i], nlist[i], brklist[i], tlist[i]) = mpc_alip.solve(s[i-1], v[i-1], output=0);
        else:
            v[i] = v[i-1];
            Clist[i]  = Clist[i-1];
            nlist[i]  = -1;
            brklist[i] = 100;
            tlist[i]  = -1;

        # construct q_d for ID-QP function
        z_d[i] = [0.0, height, theta, v[i][0]];

        if (np.isnan(Clist[i])):
            print("ERROR: Cost is nan after optimization...");
            break;

        # convert input: alip -> tpm
        u[i] = id.convert(inputs_tpm, z_d[i], q[i-1], u[i-1]);

        if (u[i] is None):
            print("ERROR: ID-QP function returned None...");
            u[i] = [0,0,0];
            break;

        if i*sim_dt > t_disturb[0] and i*sim_dt < t_disturb[1]:
            force_x = t_disturb[2];
            force_z = 0;
            link = "torso";	#options: torso, thigh, shin
            env.apply_force(link, force_x, force_z);

        # TPM simulation step
        q[i] = env.step(u[i])[0].tolist();

        # convert state: tpm -> alip
        (x_c, h_c, q_c) = CoM_tpm(q[i], inputs_tpm);
        L   = mathexp.base_momentum(q[i][:N_tpm], q[i][N_tpm:2*N_tpm])[0][0];
        L_c = mathexp.centroidal_momentum(q[i][:N_tpm], q[i][N_tpm:2*N_tpm])[0][0];

        s[i] = [x_c, L];
        z_a[i] = [x_c, h_c, q_c, L_c];

    ans = input("\nSee animation? [y/n] ");
    if (ans == 'y'):
        for i in range(Nt):
            env.set_state(np.array(q[i][0:3]), np.array(q[i][3:6]));
            env.render();
            time.sleep(0.0005);

        input("Press enter to exit program...");

    alip_results = (T, s, v, Clist, nlist, brklist, tlist);
    tpm_results  = (T, q, u, z_a, z_d);

    saveResults_alip("resultsALIP.pickle", alip_results);
    saveResults_tpm("resultsTPM.pickle", tpm_results);
