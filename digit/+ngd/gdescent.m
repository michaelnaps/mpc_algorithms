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
function [u, C, n, brk, a] = gdescent(model, P, dt, q0, u0, um, Cq, qd, arng, eps, h)
    %% Setup - Initial Guess, Cost, Gradient, and Hessian
    a = -1;
    N = length(um);
    uc = u0;
    Cc = ngd.cost(model, P, dt, q0, u0, uc, Cq, qd);
    un = uc;  Cn = Cc;

    %% Loop for Newton's Method
    count = 1;
    brk = 0;
    fprintf("Initial Iteration: %i, Cost: %.3f\n", count, Cn)
    while (Cc > eps)
        % gradient and corresponding MSE to zero
        g = ngd.cost_gradient(model, P, dt, q0, u0, uc, Cq, qd, h);
        gnorm = sqrt(sum(g.^2))/N;

        % first order optimality break according to L2-norm
        if (gnorm < eps)
            brk = 1;
            break;
        end

        % gradient descent step
        [un, ~, ~, a] = ngd.alpha_bis(model, g, P, dt, q0, u0, uc, Cq, qd, arng, eps);
%         [un, ~, ~, a] = ngd.alpha_blk(model, g, P, dt, q0, u0, uc, Cq, qd, arng);

        fprintf("Iteration: %i, Cost: %.3f, |g|: %.6f, a: %.6f\n", count, Cn, gnorm, a)

        % compute new values for cost, gradient, and hessian
        Cn = ngd.cost(model, P, dt, q0, u0, un, Cq, qd);
        Cdn = abs(Cn - Cc);
        Cnorm = sqrt(sum(Cdn.^2))/N;
        count = count + 1;

        % change in Cost break
        if (Cnorm < eps^2)
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
    fprintf("Final Iteration: %i, Cost: %.3f\n", count, Cn)
    u = un;
    C = Cn;
    n = count;
end