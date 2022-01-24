%% Function: cost()
%  Created by: Michael Napoli
%
%  Purpose: Calculate the cost for the given input over the
%   prediction horizon.
%
%  Inputs:
%   'P'   - length of prediction horizon (PH)
%   'dt'  - step-size for prediction horizon
%   'q0'  - initial state
%   'u0'  - previous input
%   'u'   - input for the gradient to be taken around
%   'c'   - coefficient of damping for each link
%   'm'   - mass at end of each pendulum
%   'L'   - length of pendulum links
%   'Cq'  - system of quadratic cost equations
%   'loc' - location of the original function call
%
%  Outputs:
%   'Cs'  - sum of the cost for each link and given input
function Cs = cost(P, dt, q0, u0, u, c, m, L, Cq, thd, loc)
    %% Cost of Constant Input
    % calculate the state over the desired prediction horizon
    qc = modeuler(P, dt, q0, u, c, m, L, loc);
    
    % sum of cost of each step of the prediction horizon
    du = (u0 - u);
    C = zeros(size(u));
    for i = 1:P+1
        C = C + Cq(thd, qc(i,:), du);
    end
    Cs = sum(C);
end