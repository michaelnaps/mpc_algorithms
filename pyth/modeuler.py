# File: modeuler.py
# Created by: Michael Napoli
# Created on: 1/17/2022
#
# Purpose: To calculate the states of a given model
#   over a predetermined prediction horizon.
def modeuler(statespace, sim_time, dt, q0, u, inputs, dt_min=1e-3):
    k = int(sim_time/dt);
    N = len(q0);

    if (dt >= dt_min):
        adj = int(dt/dt_min);
    else:
        adj = 1;

    km = k*adj;  dtm = dt/adj;
    q  = [[0 for j in range(N)] for i in range(k+1)];
    qm = [[0 for j in range(N)] for i in range(km+1)];

    q[0]  = q0;
    qm[0] = q0;
    for i in range(km):
        dq1 = statespace(qm[i], u, inputs);
        qeu = [qm[i][j] + dq1[j]*dtm for j in range(N)];
        dq2 = statespace(qeu, u, inputs);
        qm[i+1] = [qm[i][j] + 1/2*(dq1[j] + dq2[j])*dtm for j in range(N)];

        if ((i+1) % adj == 0):  q[int(i/adj+1)] = qm[i+1];

    T = [i*dt for i in range(k+1)];

    return (T, q);
