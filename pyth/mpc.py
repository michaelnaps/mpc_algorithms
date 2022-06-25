# Class: mpc.py
# Created by: Michael Napoli
# Created on: 5/31/2022


import numpy as np
import math
import time
import pickle

class system:
    def __init__(self, solver, costFunction, modelFunction, user_inputs,
                 num_inputs, num_ssvar, PH_length=1, knot_length=1, time_step=0.025,
                 appx_zero=1e-6, step_size=1e-3, max_iter=10):
        self.solver = solver;
        self.cost   = costFunction;
        self.model  = modelFunction;
        self.inputs = user_inputs;
        self.u_num  = num_inputs;
        self.q_num  = num_ssvar;
        self.PH     = PH_length;
        self.k      = knot_length;
        self.dt     = time_step;
        self.zero   = appx_zero;
        self.h      = step_size;
        self.n_max  = max_iter;

        self._dt_min = 1e-3;

        self._alpha = 1;
        self._bkl_shrink = 0.1;
        self._a_method = 'none';

    def setModelInputs(self, user_inputs):
        self.inputs = user_inputs;
        return 1;

    def setMinTimeStep(self, min_time_step):
        self._dt_min = min_time_step;
        return 1;

    def getMinTimeStep(self):
        return self._dt_min;

    def setAlpha(self, a):
        self._alpha = a;
        return 1;

    def getAlpha(self):
        return self._alpha;

    def setAlphaMethod(self, method, shrink=0.1):
        self._a_method = method;
        self._bkl_shink = shrink;
        return 1;

    def getAlphaMethod(self):
        return self._a_method;

    def solve(self, q0, uinit, output=0):

        if (self.solver == 'nno'):
            t = time.time();
            (u, C, n, brk) = self.nno(q0, uinit, output);
            elapsed = 1000*(time.time() - t);
        if (self.solver == 'ngd'):
            t = time.time();
            (u, C, n, brk) = self.ngd(q0, uinit, output);
            elapsed = 1000*(time.time() - t);

        return (u, C, n, brk, elapsed);

    def nno(self, q0, uinit, output=0):
        # MPC constants initialization
        N      = self.u_num;
        P      = self.PH;
        eps    = self.zero;
        imax   = self.n_max;
        inputs = self.inputs;

        # loop variable setup
        uc = uinit;
        q  = self.simulate(q0, uc);
        Cc = self.cost(self, q, uc, inputs);
        un = uc;  Cn = Cc;

        if output:
            print("Opt. Start:");
            print("Initial Cost: ", Cc);

        count = 0;
        brk = -2*np.isnan(Cc);
        while (Cc > eps):
            # calculate the gradient around the current input
            g = self.gradient(q0, uc);
            gnorm = np.sqrt(np.sum([g[i]**2 for i in range(N)]));

            # check if gradient-norm is an approx. of zero
            if (gnorm < eps):
                brk = 1;
                break;

            # calculation the hessian around the current input
            H = self.hessian(q0, uc);

            # calculate the next iteration of the input
            udn = np.linalg.lstsq(H, g, rcond=None)[0];
            un = [uc[i] - udn[i] for i in range(P*N)];

            # simulate and calculate the new cost value
            q  = self.simulate(q0, un);
            Cn = self.cost(self, q, un, inputs);
            count += 1;  # iterate the loop counter

            if (np.isnan(Cn)):
                brk = -2;
                break;

            if output:
                # print("Gradient:  ", g);
                print("|g|:          ", gnorm);
                print("New Cost:     ", Cn);
                print("New Input: ");
                for i in range(0, N*P, N):
                    print("              ", [un[i+j] for j in range(N)]);

            # break conditions
            if (count == imax):
                brk = -1;
                break;

            # update loop variables
            uc = un;  Cc = Cn;

        return (un, Cn, count, brk);

    def ngd(self, q0, uinit, output=0):
        # MPC constants initialization
        N      = self.u_num;
        P      = self.PH;
        eps    = self.zero;
        imax   = self.n_max;
        inputs = self.inputs;

        # step size coefficient choice
        alpha = self._alpha;
        a_method = self._a_method;

        # loop variable setup
        uc = uinit;
        q  = self.simulate(q0, uc);
        Cc = self.cost(self, q, uc, inputs);
        un = uc;  Cn = Cc;

        if output:
            print("Opt. Start:");
            print("Initial Cost: ", Cc);

        count = 1;
        brk = -2*np.isnan(Cc);
        while (Cc > eps):
            # calculate the gradient around the current input
            g = self.gradient(q0, uc);
            gnorm = np.sqrt(np.sum([g[i]**2 for i in range(N)]));

            # check if gradient-norm is an approx. of zero
            if (gnorm < eps):
                brk = 1;
                break;

            # calculate the next iteration of the input
            if (a_method == "bkl"):
                (un, Cn, _, _, _) = self.alpha_bkl(g, Cc, q0, uc);
            else:
                un = [uc[i] - alpha*g[i] for i in range(P*N)];
                q  = self.simulate(q0, un);
                Cn = self.cost(self, q, un, inputs);

            count += 1;  # iterate the loop counter

            if (np.isnan(Cn)):
                brk = -2;
                break;

            if output:
                # print("Gradient:  ", g);
                print("|g|:          ", gnorm);
                print("New Cost:     ", Cn);
                print("New Input: ");
                for i in range(0, N*P, N):
                    print("              ", [un[i+j] for j in range(N)]);

            # break conditions
            if (count > imax):
                brk = -1;
                break;

            if (np.abs(Cc - Cn) < eps):
                brk = 2;
                break;

            # update loop variables
            uc = un;  Cc = Cn;

        return (un, Cn, count, brk);

    def gradient(self, q0, u, rownum=1):
        # variable setup
        N = self.u_num*self.PH;
        h = self.h;
        g = [0 for i in range(N)];
        inputs = self.inputs;

        for i in range(rownum-1, N):
            un1 = [u[j] - (i==j)*h for j in range(N)];
            up1 = [u[j] + (i==j)*h for j in range(N)];

            qn1 = self.simulate(q0, un1);
            qp1 = self.simulate(q0, up1);

            Cn1 = self.cost(self, qn1, un1, inputs);
            Cp1 = self.cost(self, qp1, up1, inputs);

            g[i] = (Cp1 - Cn1)/(2*h);

        return g;

    def hessian(self, q0, u):
        # variable setup
        N = self.u_num*self.PH;
        h = self.h;
        H = [[0 for i in range(N)] for j in range(N)];

        for i in range(N):
            un1 = [u[j] - (i==j)*h for j in range(N)];
            up1 = [u[j] + (i==j)*h for j in range(N)];

            gn1 = self.gradient(q0, un1, i);
            gp1 = self.gradient(q0, up1, i);

            # enforce symmetry
            for j in range(i, N):
                H[i][j] = (gp1[j] - gn1[j])/(2*h);
                if (i != j):  H[j][i] = H[i][j];

        return H;

    def alpha_bkl(self, g, C, q0, uc):
        P = self.PH;
        N = self.u_num;
        eps = self.zero;
        a = self._alpha;
        w = self._bkl_shrink;
        inputs = self.inputs;

        count = 0;
        brk = -1;
        while ((count != 1000) & (a > eps)):
            ubkl = [uc[i] - a*g[i] for i in range(P*N)];
            q  = self.simulate(q0, ubkl);
            Cbkl = self.cost(self, q, ubkl, inputs);
            count += 1;

            if (Cbkl < C):
                brk = 0;
                break;

            a *= (1 - w);

        return (ubkl, Cbkl, a, count, brk);

    def simulate(self, q0, u):
        # mpc variables
        N  = self.q_num;
        Nu = self.u_num;
        P  = self.PH;

        # reshape input variable
        uc = np.reshape(u, [P, Nu]);

        # Cost of each input over the designated windows
        # simulate over the prediction horizon and sum cost
        q = [[0 for i in range(2*N)] for j in range(P+1)];
        q[0] = q0;
        for i in range(P):
            q[i+1] = self.modeuler(q[i], uc[i])[1][-1];

        return q;

    def modeuler(self, q0, u, knot_length=-1):
        N  = self.q_num;
        P  = self.PH;
        dt = self.dt;
        dt_min = self._dt_min;
        inputs = self.inputs;

        if (knot_length == -1):  k = self.k;
        else:  k = knot_length;

        if (dt >= dt_min):  adj = int(dt/dt_min);
        else:  adj = 1;

        km = k*adj;  dtm = dt/adj;
        q  = [[0 for j in range(2*N)] for i in range(k+1)];
        qm = [[0 for j in range(2*N)] for i in range(km+1)];

        q[0]  = q0;
        qm[0] = q0;
        for i in range(km):
            dq1 = self.model(qm[i], u, inputs);
            qeu = [qm[i][j] + dq1[j]*dtm for j in range(N)];
            dq2 = self.model(qeu, u, inputs);
            qm[i+1] = [qm[i][j] + 1/2*(dq1[j] + dq2[j])*dtm for j in range(N)];

            if ((i+1) % adj == 0):  q[int(i/adj+1)] = qm[i+1];

        T = [i*dt for i in range(k+1)];

        return (T, q);

    def sim_root(self, sim_time, q0, u0, update=0, output=0):
        # mpc variables
        N  = self.u_num;
        P  = self.PH;
        dt = self.dt;

        # simulation time variables
        Nt = int(sim_time/dt+1);
        T = [i*dt for i in range(Nt)];

        # state matrices declarations
        qlist = [0 for i in range(Nt)];
        qlist[0] = q0;

        # return variables
        ulist = [0 for i in range(Nt)];
        Clist = [0 for i in range(Nt)];
        nlist = [0 for i in range(Nt)];
        brklist = [0 for i in range(Nt)];
        tlist = [0 for i in range(Nt)];

        ulist[0] = u0;

        for i in range(1,Nt):
            if output:  print("\nTime: %0.3f" % (T[i]));

            opt_results = self.solve(qlist[i-1], ulist[i-1], output);

            ulist[i]   = opt_results[0];
            Clist[i]   = opt_results[1];
            nlist[i]   = opt_results[2];
            brklist[i] = opt_results[3];
            tlist[i]   = opt_results[4];

            if output:  print("Elapsed Time:\n              ", tlist[i]);

            qlist[i] = self.modeuler(qlist[i-1], ulist[i][:N], 1)[1][1];

            if (update != 0):  self.inputs = update(self, qlist[i], ulist[i]);

        return (T, qlist, ulist, Clist, nlist, brklist, tlist);
