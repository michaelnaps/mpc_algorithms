%% Optimization Algorithm II
%  Method: Nonlinear Newton's (NN) Optimization Method
%  Created by: Michael Napoli
%
%  Purpose: To optimize and solve a given system
%   of n-link pendulum models using model predictive
%   control (MPC) and the NN algorithm with the
%   necessary constraints.
%
%  Inputs:
%   'P'   - length of prediction horizon (PH)
%   'dt'  - step-size for prediction horizon
%   'q0'  - initial state
%   'u0'  - initial guess
%   'um'  - maximum allowable torque values
%   'c'   - coefficient of damping for each link
%   'm'   - mass at end of each pendulum
%   'L'   - length of pendulum links
%   'Cq'  - system of quadratic cost equations
%   'eps' - acceptable error (for breaking)
%
%  Outputs:
%   'u'   - inputs for each applicable joint
%   'C'   - cost of the links for each window of PH
%   'n'   - number of iterations needed
%   'brk' - loop break code
%        -1 -> iteration break (1000)
%         0 -> zero cost break
%         1 -> first order optimality
%         2 -> change in input break

% function [u, C, n, brk] = newtons(P, dt, q0, u0, um, Cq, thd, eps);
% function [u] = newtons(P, dt, q0, u0, um, Cq, qd, eps, input, model, logger)
function [u] = newtons(input, model, ~, q0, logger) 
    %% Constant Parameters
    N = length(q0)/2;
    P   = input.Params.P;
    dt  = input.Params.dt;
    um  = input.Params.um;
    Cq  = input.Params.Cq;
    qd  = input.Params.qd;
    eps = input.Params.eps;

    if (isempty(fieldnames(logger.calc)))
        u0 = zeros(N, 1);
    else
        u0 = logger.calc.torque;
    end

    %% Setup - Initial Guess, Cost, Gradient, and Hessian
    uc = u0;
    Cc = nno.cost(P, dt, q0, u0, uc, Cq, qd, model);
    un = uc;  Cn = Cc;

    %% Loop for Newton's Method
    count = 1;
    brk = 0;
    while (Cc > eps)
        % gradient and hessian of the current input
        g = nno.cost_gradient(P, dt, q0, u0, uc, Cq, qd, 1e-3, model);
        H = nno.cost_hessian(P, dt, q0, u0, uc, Cq, qd, 1e-3, model);

        % calculate and add the next Newton's step
        un = uc - H\g;

        % compute new values for cost, gradient, and hessian
        Cn = nno.cost(P, dt, q0, u0, un, Cq, qd, model);
        udn = abs(un - uc);
        count = count + 1;

        % first order optimality break
        if (sum(g < eps) == N)
            brk = 1;
            break;
        end

        % change in input break
        if ((udn < eps) == N)
            brk = 2;
            break;
        end

        % maximum iteration break
        if (count == 1000)
            brk = -1;
            break;
        end

        % update current variables for next iteration
        uc = un;
    end
        
    % check boundary constraints
    for i = 1:N
        if (un(i) > um)
            un(i) = um;
        elseif (un(i) < -um)
            un(i) = -um;
        end
    end

    %% Return Values for Input, Cost, and Iteration Count
    u = un;
    C = Cn;
    n = count;

    if nargin > 3
        calc = logger.calc;

        calc.torque = u;
        calc.cost = C;
        calc.iterations = n;
        calc.break = brk;
        
        calc.qd = qd;
        calc.dqd = dqd;

        logger.calc = calc;
    end
end