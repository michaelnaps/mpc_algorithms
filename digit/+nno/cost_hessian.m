%% Function: cost_hessian()
%  Method: 2-point Central Finite Difference Method (CFDM)
%  Created by: Michael Napoli
%
%  Purpose: Calculate the Hessian matrix for an n-link
%   pendulum using the subsequent gradient over the prediction
%   horizon.
%
%  Inputs:
%   'P'   - length of prediction horizon (PH)
%   'dt'  - step-size for prediction horizon
%   'q0'  - initial state
%   'u0'  - previous input
%   'u'   - input for the hessian to be taken around
%   'c'   - coefficient of damping for each link
%   'm'   - mass at end of each pendulum
%   'L'   - length of pendulum links
%   'Cq'  - system of quadratic cost equations
%   'h'   - step size for change in input
%   'loc' - location of the original function call
%
%  Outputs:
%   'H'   - hessian matrix (d2C/du2)
function [H] = cost_hessian(model, P, dt, q0, u0, u, Cq, qd, h)
    %% Setup
    N = length(u);
    H = zeros(N);
    
    %% Finite Difference Method (H = hessian)
    for i = 1:N
        un1 = u;
        up1 = u;

        un1(i) = u(i) - h;
        up1(i) = u(i) + h;

        Hn1 = nno.cost_gradient(model, P, dt, q0, u0, un1, Cq, qd, h);
        Hp1 = nno.cost_gradient(model, P, dt, q0, u0, up1, Cq, qd, h);

        Hn = (Hp1 - Hn1)/(2*h);

%         un2 = u;
%         up2 = u;
% 
%         un2(i) = u(i) - 2*h;
%         up2(i) = u(i) + 2*h;
% 
%         Hn2 = cost_gradient(model, P, dt, q0, u0, un2, Cq, qd, h, model);
%         Hp2 = cost_gradient(model, P, dt, q0, u0, up2, Cq, qd, h, model);
% 
%         Hn = (Hn2 - 8*Hn1 + 8*Hp1 - Hp2)/(12*h);
        
        H(i,:) = Hn';
    end
end