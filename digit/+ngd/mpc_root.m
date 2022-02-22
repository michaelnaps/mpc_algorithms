function [u] = mpc_root(input, model, Tc, q0, logger)
    %% Constant Parameters initializations
    P     = input.Params.P;
    dt    = input.Params.dt;
    um    = input.Params.um;
    Cq    = input.Params.Cq;
    qd    = input.Params.qd;
    dqd   = input.Params.dqd;
    arng  = input.Params.arng;
    eps   = input.Params.eps;
    h     = input.Params.stepSize;
    a_ind = input.Params.a_ind;
    testID = input.Params.testID;

    if (isempty(fieldnames(logger.calc)))
        % u0 = zeros(N,1);
        u0 = input.Params.ui;
    else
        u0 = logger.calc.torque;
    end
    
    %% Run Optimization Algorithm
    tic;
    [u, C, n, brk] = ngd.gdescent(model, P, dt, q0, u0, um, Cq, qd, arng, eps, h);
    t = toc;
    
%     save_newton((testID + "_data.csv"), [u', C, n, brk]);
    fprintf("State Calculated: t = %.6f\nOpt Time: %.3f [s], Iterations: %i, Break: %i\n\n", Tc, t, n, brk);

    %% Log Data and Return
    if nargin > 4
        calc = logger.calc;

        calc.torque = u;
        calc.cost = C;
        calc.iterations = n;
        calc.break = brk;
        calc.nno_time = t;
        
        calc.act_ind = a_ind;
        calc.qd = qd;
        calc.dqd = dqd;

        logger.calc = calc;
    end
    
end
