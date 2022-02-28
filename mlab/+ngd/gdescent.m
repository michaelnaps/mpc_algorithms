%% Optimization Algorithm III
%  Method: Gradient Descent with Alpha Search
%  Created by: Michael Napoli
%
%  Purpose: To optimize and solve a given system
%   of n-link pendulum models using model predictive
%   control (MPC) and the NGD algorithm with the
%   necessary constraints.
%
%  Inputs:
%   'P'    - length of prediction horizon (PH)
%   'dt'   - step-size for prediction horizon
%   'q0'   - initial state
%   'u0'   - initial guess
%   'um'   - maximum allowable torque values
%   'c'    - coefficient of damping for each link
%   'm'    - mass at end of each pendulum
%   'L'    - length of pendulum links
%   'Cq'   - system of quadratic cost equations
%   'qd'   - list of desired angles
%   'arng' - maximum and minimum allowable step sizes
%   'eps'  - acceptable error (for breaking)
%   'h'    - step size of gradient calculation function
%
%  Outputs:
%   'u'   - inputs for each applicable joint
%   'C'   - cost of the links for each window of PH
%   'n'   - number of iterations needed
%   'brk' - loop break code
%        -1 -> iteration break (1000)
%         0 -> zero cost break
%         1 -> first order optimality (L2 norm of gradient)
%         2 -> change in input break (L2 norm of input changes)
function [u, C, n, brk, a] = gdescent(P, dt, q0, u0, um, c, m, L, Cq, qd, arng, eps, h)
    %% Setup - Initial Guess, Cost, Gradient, and Hessian
    a = -1;
    N = length(um);
    uc = u0;
    Cc = ngd.cost(P, dt, q0, u0, uc, c, m, L, Cq, qd, " NN Initial Cost ");
    un = uc;  Cn = Cc;

    %% Loop for Newton's Method
    count = 1;
    brk = 0;
    while (Cc > eps)
        % gradient and corresponding MSE to zero
        g = ngd.cost_gradient(P, dt, q0, u0, uc, c, m, L, Cq, qd, h, " Main Loop Gradient ");
        gnorm = sqrt(sum(g.^2))/N;

        % first order optimality break according to L2-norm
        if (gnorm < eps)
            brk = 1;
            break;
        end

        % gradient descent step
%         [un, ~, ~, a] = ngd.alpha_bis(g, P, dt, q0, u0, uc, c, m, L, Cq, qd, arng, eps);
        [un, ~, ~, a] = ngd.alpha_blk(g, Cc, P, dt, q0, u0, uc, c, m, L, Cq, qd, arng, eps);

        % compute new values for cost, gradient, and hessian
        Cn = ngd.cost(P, dt, q0, u0, un, c, m, L, Cq, qd, " NN Main Loop Cost ");
        udn = abs(un - uc);
        unorm = sqrt(sum(udn.^2))/N;
        count = count + 1;

        % change in input break
        if (unorm < eps)
            brk = 2;
            break;
        end

        % maximum iteration break
        if (count == 1000)
            fprintf("ERROR: Iteration break. (%i)\n", count)
            brk = -1;
            break;
        end

        % update current variables for next iteration
        uc = un;  Cc = Cn;
    end
        
    % check boundary constraints
    for i = 1:N
        if (un(i) > um(i))
            un(i) = um(i);
        elseif (un(i) < -um(i))
            un(i) = -um(i);
        end
    end

    %% Return Values for Input, Cost, and Iteration Count
    u = un;
    C = Cn;
    n = count;
end