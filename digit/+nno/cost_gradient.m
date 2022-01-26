%% Function: cost_gradient()
%  Method: 2-point Central Finite Difference Method (CFDM)
%  Created by: Michael Napoli
%
%  Purpose: Calculate the gradient vector for an n-link
%   pendulum using the subsequent gradient over the prediction
%   horizon.
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
%   'h'   - step size for change in input
%   'loc' - location of the original function call
%
%  Outputs:
%   'g'   - gradient vector (dC/du)
function [g] = cost_gradient(P, dt, q0, u0, u, Cq, qd, h, model, a_ind)
    %% Setup
    N = length(u);
    g = zeros(size(u));

    %% Finite Difference Method (g = gradient)
    for i = 1:N
        un1 = u;
        up1 = u;

        un1(i) = u(i) - h;
        up1(i) = u(i) + h;
        
        Cn1 = nno.cost(P, dt, q0, u0, un1, Cq, qd, model, a_ind);
        Cp1 = nno.cost(P, dt, q0, u0, up1, Cq, qd, model, a_ind);
        
        gn = (Cp1 - Cn1)/(2*h);

%         up2 = u;
%         un2 = u;
% 
%         un2(i) = u(i) - 2*h;
%         up2(i) = u(i) + 2*h;
% 
%         Cn2 = nno.cost(P, dt, q0, u0, un2, c, m, L, Cq, thd, 'Gradient u(i-2)');
%         Cp2 = nno.cost(P, dt, q0, u0, up2, c, m, L, Cq, thd, 'Gradient u(i+2)');
% 
%         gn = (Cn2 - 8*Cn1 + 8*Cp1 - Cp2)/(12*h);

        g(i) = gn;
    end
end

