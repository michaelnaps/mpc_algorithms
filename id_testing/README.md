Using Python 3.6, run the file titled: root_id.py

The ID-QP function is located in the folder titles: models, along with the
model library for the 3-link model.

There are some simple plotting functions located in the statespace library if
you would like to see inputs and state trends.

NOTE: The qpsolvers package is required to run the ID-QP optimizer, and can be
installed with the following line in the command window (if using conda and inside
the necessary conda environment shell):
   conda install -c conda-forge qpsolvers

If you would like to see the variable components calculated from the ID-QP function,
the function takes an optional input titled "output". If this is set to 1, then the
components will be printed to the command window during runtime. See example below.
   u_3link[i] = id.convert(inputs_3link, q_desired[i], q_3link[i]);
   vs.
   u_3link[i] = id.convert(inputs_3link, q_desired[i], q_3link[i], 1);

I have compared the modeuler() function to scipy.integrate.solve_ivp() for multiple
sets of trends, so I am fairly certain it is not the issue.

The following list of variables are the MATLAB to Python ID-QP components, I did not
use the same naming convention as the Professor. For this reason I am writing every
variable's equivalent here...

MATLAB = Python
q = x
dq = dx
M = M
H = E

y_d = q_d
dy_d = dq_d
ddy_d = ddq_d
L_c_des = Lc_d

y_a = q_a
dy_a = dq_a

Jy_a = J_a
dJy_a = dJ_a

kp = kp
kd = kd
mu = u_PD

L_c = Lc
JL_c = J_Lc
dJL_c = dJ_Lc

A = J
b = u

Hmat = H
bmat = g
Aeq = A
beq = b
lb = lb
ub = ub
Aineq = G
bineq = h
