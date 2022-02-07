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
function Cs = cost(P, dt, q0, u0, u, Cq, qd, model)
    %% Cost of Constant Input
    % calculate the state over the desired prediction horizon
    adj = 10;
    % qc = modeuler(P, dt, q0, u, model);
    [~, qc] = ode45(@(t,q) statespace_digit(q, u, model), 0:dt/adj:P*dt, q0);

    % sum of cost of each step of the prediction horizon
    N = length(q0)/2;
    C = zeros(N,1);
    for i = 1:length(qc(:,1))
        if (rem(i,adj) == 0)
            for j = 1:N
                C(j) = C(j) + Cq([qd(j), 0.0], [qc(i,j), qc(i,N+j)]);
            end
        end
    end
    Cs = sum(C);
end