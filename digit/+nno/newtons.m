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
function [u, C, n, brk] = newtons(P, dt, q0, u0, um, Cq, qd, eps, model, a_ind)
    %% Setup - Initial Guess, Cost, Gradient, and Hessian
    N = length(q0)/2;
    uc = u0;
    Cc = nno.cost(P, dt, q0, u0, uc, Cq, qd, model, a_ind);
    un = uc;  Cn = Cc;

    %% Loop for Newton's Method
    count = 1;
    brk = 0;
    while (Cc > eps)
        % gradient and hessian of the current input
        g = nno.cost_gradient(P, dt, q0, u0, uc, Cq, qd, 1e-3, model, a_ind);
        H = nno.cost_hessian(P, dt, q0, u0, uc, Cq, qd, 1e-3, model, a_ind);

        % calculate and add the next Newton's step
        un = uc - H\g;

        % compute new values for cost, gradient, and hessian
        Cn = nno.cost(P, dt, q0, u0, un, Cq, qd, model, a_ind);
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