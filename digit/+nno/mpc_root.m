function [u] = mpc_root(input, model, ~, q0, logger)
    %% Constant Parameters initializations
    N     = length(q0)/2;
    P     = input.Params.P;
    dt    = input.Params.dt;
    um    = input.Params.um;
    Cq    = input.Params.Cq;
    qd    = input.Params.qd;
    dqd   = input.Params.dqd;
    eps   = input.Params.eps;
    a_ind = input.Params.a_ind;

    if (isempty(fieldnames(logger.calc)))
        u0 = zeros(N,1);
    else
        u0 = logger.calc.torque;
    end
    
    %% Run Optimization Algorithm
    tic;
    [u, C, n, brk] = nno.newtons(P, dt, q0, u0, um, Cq, qd, eps, model, a_ind);
    t = toc;
    fprintf("runtime: %.3f - iterations: %i\n", t, n);

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