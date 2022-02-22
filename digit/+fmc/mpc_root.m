function [u] = mpc_root(input, model, Tc, q0, logger)
    %% Constant Parameters initializations
    N     = length(q0)/2;
    P     = input.Params.P;
    dt    = input.Params.dt;
    Cq    = input.Params.Cq;
    qd    = input.Params.qd;
    dqd   = input.Params.dqd;
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
    options = optimset("display", "iter", 'tolfun', 1e-3);
    tic;
    [u, C, brk, output] = fminunc(@(u) fmc.cost(P, dt, q0, u0, u, Cq, qd, model), q0(a_ind), options);
    t = toc;

%     TF = save_newton(testID + "_data.csv", [Tc, u', C, n, brk]);

%     fprintf("Initial Guess:\n")
%     for i = 1:length(u0)
%         fprintf("\t%.10f\n", u0(i))
%     end
%     fprintf("Final Input:\n")
%     if (u ~= u0)
%         for i = 1:length(u)
%             fprintf("\t%.10f\n", u(i))
%         end
%     end
    
    n = output.iterations;
    fprintf("State Calculated: t = %.6f\nOpt Time: %.3f [s], Iterations: %i, Break: %i\n\n", Tc, t, n, brk);

    %% Log Data and Return
    if nargin > 4
        calc = logger.calc;

        calc.torque = u;
        calc.cost = C;
        calc.iter = n;
        calc.break = brk;
        calc.nno_time = t;
        
        calc.act_ind = a_ind;
        calc.qd = qd;
        calc.dqd = dqd;

        logger.calc = calc;
    end
    
end