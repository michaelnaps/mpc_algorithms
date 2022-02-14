%% Optimization Algorithm II
%  Method: Nonlinear Newton's (NN) Optimization Method
%  Created by: Michael Napoli
%
%  Purpose: To optimize and solve a given robotic
%   system using a Model Predictive Control architecture
%   in combination with Newton's Optimization.
%
%  Inputs:
%   'P'     - length of prediction horizon (PH)
%   'dt'    - step-size for prediction horizon
%   'q0'    - initial state
%   'u0'    - initial guess
%   'um'    - maximum allowable torque values
%   'Cq'    - system of quadratic cost equations
%   'qd'    - list of desired states for joints
%   'eps'   - acceptable error (for breaking)
%   'model' - robot model created using FROST software
%   'a_ind' - indices for controlled joints         
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
function [u, C, n, brk] = newtons(P, dt, q0, u0, um, Cq, qd, eps, h, model)
    %% Setup - Initial Guess, Cost, Gradient, and Hessian
    N = length(q0)/2;
    uc = u0;
    Cc = nno.cost(P, dt, q0, u0, uc, Cq, qd, model);
    un = uc;  Cn = Cc;

    %% Loop for Newton's Method
    count = 1;
    brk = 0;
    while (Cc > eps)
        % gradient and corresponding MSE to zero
        g = nno.cost_gradient(P, dt, q0, u0, uc, Cq, qd, h, model);
        gnorm = sqrt(sum(g.^2))/N;

        % first order optimality break according to L2-norm
        if (gnorm < eps)
            brk = 1;
            break;
        end

        % calculate the Hessian matrix and corresponding Newton's step
        H = nno.cost_hessian(P, dt, q0, u0, uc, Cq, qd, h, model);
        un = uc - H\g;
%         un = uc - alph*g;  % alternative: gradient descent

        % compute new values for cost, gradient, and hessian
        Cn = nno.cost(P, dt, q0, u0, un, Cq, qd, model);
        udn = abs(un - uc);  Cdn = abs(Cn - Cc);
        count = count + 1;

        fprintf("Initial Cost: %.3f\tCurrent cost: %.3f\tChange in cost: %.6f\tGradient Norm: %.6f\n", Cc, Cn, Cdn, gnorm)

%         % change in input break based on MSE of udn to zero
%         unorm = sqrt(sum(udn.^2))/N;
%         if (unorm < eps)
%             brk = 2;
%             break;
%         end

        % change in cost break
        if (Cdn < eps)
            brk = 3;
            break;
        end

        % maximum iteration break
        if (count == 30)
            brk = -1;
            break;
        end
        
        % update current variables for next iteration
        uc = un;  Cc = Cn;
    end
        
    % check boundary constraints
    for i = 1:length(un)
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
end