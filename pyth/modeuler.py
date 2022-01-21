# File: modeuler.py
# Created by: Michael Napoli
# Created on: 1/17/2022
#
# Purpose: To calculate the states of a given statespace
#   over a predetermined prediction horizon.

def statespace(q, u, c, m, L, jNum):
    return [i for i in range(2*jNum)];

def modeuler(P, dt, q0, u, c, m, L, jNum):
    Pm = P*10;  dtm = dt/10;
    q = [[0 for j in range(2*jNum)] for i in range(P+1)];
    qm = [[0 for j in range(2*jNum)] for i in range(Pm+1)];

    q[0] = q0;
    qm[0] = q0;
    for i in range(Pm):
        dq1 = statespace(qm[i], u, c, m, L, jNum);
        qeu = [(qm[i][j] + dq1[j]*dtm) for j in range(2*jNum)];
        qeu = [(i*j) for j in range(2*jNum)];
        dq2 = statespace(qeu, u, c, m, L, jNum);
        qm[i+1] = [(qm[i][j] + 1/2*(dq1[j] + dq2[j])*dtm) for j in range(2*jNum)];

        if (i % 10 == 0):
            q[int(i/10+1)] = qm[i+1];

    return q;

print(modeuler(4, 0.025, [1,2,3,4,5,6], 1, 1, 1, 1, 3));
