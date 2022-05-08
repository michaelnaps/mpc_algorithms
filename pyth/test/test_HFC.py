import sys

sys.path.insert(0, '/home/michaelnaps/prog/mpc_algorithms/pyth/.');
sys.path.insert(0, 'models/.');

import nno
from statespace_lapm import *
import matplotlib.pyplot as plt

def Cq(qd, q):
    Cq = [
        100*(qd[0] - q[0])**2 +  1*(qd[2] - q[2])**2,
        10*(qd[1] - q[1])**2 +    (qd[3] - q[3])**2
    ];

return np.sum(Cq);

def Cu(u, du):
    umax = [60, 150];

    Cu = [
        1e-5*(du[0])**2,# - np.log(umax[0]**2 - u[0]**2) + np.log(umax[0]**2),
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
        0
    ];

    return np.sum(Ccmp);

class mpc_var:
    sim_time     = 2;
    model        = statespace_lapm;
    cost_state   = Cq;
    cost_input   = Cu;
    cost_CMP     = Ccmp;
    PH_length    = 10;
    knot_length  = 2;
    time_step    = 0.025;
    appx_zero    = 1e-6;
    step_size    = 1e-3;
    num_ssvar    = 2;
    num_inputs   = 2;
    input_bounds = [1000 for i in range(num_inputs*PH_length)];
    des_config   = [0, 0, 0, 0];
    hessian      = 0;

class inputs:
    # Constants and State Variables
    num_joints           = 3;
    gravity_acc          = -9.81;
    damping_coefficients = [0];
    joint_masses         = [80];
    link_lengths         = [2.0];
    CP_maxdistance       = 0.1;

q0 = [0-0.05, 0, 0, 0];
u0 = [0 for i in range(mpc_var.num_inputs*mpc_var.PH_length)];

N_test = 10;
t_hessian = [0 for i in range(N_test)];
t_symmetric = [0 for i in range(N_test)];
p_improvement = [0 for i in range(N_test)];
iter = [i for i in range(N_test)];

for i in range(N_test):
    mpc_var.PH_length = i+1;
    u0 = [0 for k in range(mpc_var.num_inputs*mpc_var.PH_length)];

    mpc_var.hessian = 0;
    mpc_results = nno.mpc_root(mpc_var, q0, u0, inputs, 1);
    t_hessian[i] = np.mean(mpc_results[6]);

    mpc_var.hessian = 1;
    mpc_results = nno.mpc_root(mpc_var, q0, u0, inputs, 1);
    t_symmetric[i] = np.mean(mpc_results[6]);

    p_improvement[i] = 100*(t_symmetric[i] - t_hessian[i])/t_hessian[i];

fig, runTimePlot = plt.subplots();
runTimePlot.plot(iter, t_hessian, label="Discrete Function");
runTimePlot.plot(iter, t_symmetric, label="Symmetric Funtion");
runTimePlot.set_title("Symmetric vs. Discrete Hessian Comparison");
runTimePlot.set_ylabel("Average Calc. Time [ms]");
runTimePlot.set_xlabel("PH Length [n]");
runTimePlot.legend();
runTimePlot.grid();

fig, PIplot = plt.subplots();
PIplot.plot(iter, p_improvement, label="Symmetric Funtion");
PIplot.set_title("Percent Improvement Between Hessian Functions");
PIplot.set_ylabel("PI [%]");
PIplot.set_xlabel("PH Length [n]");
PIplot.grid();

plt.show();
