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
function [u, C, n, brk] = gdescent(model, P, dt, q0, u0, um, Cq, qd, eps, h, alph0)
    %% Setup - Initial Guess, and Cost
    maxAlph = alph0;
    minAlph = 1e-3;
    N = length(q0)/2;
    uc = u0;
    Cc = ngd.cost(P, dt, q0, u0, uc, Cq, qd, model);
    un = uc;  Cn = Cc;

    %% Loop for Newton's Method
    count = 1;
    brk = 0;
    while (Cc > eps)
        % gradient and corresponding MSE to zero
        g = ngd.cost_gradient(P, dt, q0, u0, uc, Cq, qd, h, model);
        gnorm = sqrt(sum(g.^2))/N;

        % first order optimality break according to L2-norm
        if (gnorm < eps)
            brk = 1;
            break;
        end

        % calculate next iteration using gradient descent
        %   and backtracking line search
        %   if alpha falls below minimum, break
        a = maxAlph;  ua = uc - a*g;   
        b = minAlph;  ub = uc - b*g;
        Ca = ngd.cost(P, dt, q0, u0, ua, Cq, qd, model);
        Cb = ngd.cost(P, dt, q0, u0, ub, Cq, qd, model);
        while (1)
            uave = uc - (a + b)/2*g;
            Cave = ngd.cost(P, dt, q0, u0, uave, Cq, qd, model);
            % if new cost is less than previous cost, break
            if (Ca < Cb)
                b = (a + b)/2;
                Cb = Cave;
            else
                a = (a + b)/2;
                Cb = Cave;
            end

            if (Cave < eps || abs(Ca - Cb) < eps)
                break;
            end
        end

        % compute new values for cost, gradient, and hessian
        Cdn = abs(Cn - Cc);
        count = count + 1;

        fprintf("Initial Cost: %.3f\tCurrent cost: %.3f\tChange in cost: %.6f\tGradient Norm: %.6f\tAlpha: %.6f\n", Cc, Cn, Cdn, gnorm, (a + b)/2)

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
        maxAlph = alph0;  % reset alpha guess
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
