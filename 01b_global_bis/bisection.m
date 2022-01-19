%% Optimization Algorithm I
%  Method: Parallel Bisection Search (PBS)
%  Created by: Michael Napoli
%
%  Purpose: To optimize and solve a given system
%   of n-link pendulum models using model predictive
%   control (MPC) and the PBS algorithm with the
%   necessary constraints.
%
%  Inputs:
%   'P'   - length of prediction horizon (PH)
%   'dt'  - step-size for prediction horizon
%   'q0'  - initial state
%   'um'  - maximum allowable torque values
%   'c'   - coefficient of damping for each link
%   'm'   - mass at end of each pendulum
%   'L'   - length of pendulum links
%   'Cq'  - system of quadratic cost equations
%   'eps' - acceptable error (for breaking)
%
%  Outputs:
%   'u' - inputs for each applicable joint
%   'C' - cost of the links for each window of PH
%   'n' - number of iterations needed

function [u, C, n] = bisection(q0)
    %% Global Variables
    global um eps;

    %% Setup
    ua = -um;
    ub =  um;
    uave = zeros(size(um));
    du = Inf(size(um));
    
    Ca = cost(q0, ua);
    Cb = cost(q0, ub);
    Cave = cost(q0, uave);
    
    %% Optimization Loop
    count = 1;
    while (sum(Cave > eps) > 0)
        
        % update boundary variables
        for i = 1:length(uave)
            
            if (du(i) < eps)
                break;
            end

            if(Ca(i) < Cb(i))
                ub(i) = uave(i);
                Cb(i) = Cave(i);
                du(i) = abs(ua(i)-uave(i));
            else
                ua(i) = uave(i);
                Ca(i) = Cave(i);
                du(i) = abs(ub(i)-uave(i));
            end
            
        end
        
        % check change in input
        if (sum(du < eps) == length(du))
            break;
        end
        
        % update center inputs and cost
        uave = (ua + ub) ./ 2;
        Cave = cost(q0, uave);
        count = count + 1;
        
        % iteration check
        if (count == 1000)
            break;
        end

    end
    
    %% Post-Loop Processes/Checks
    if (count == 1000)
        fprintf("ERROR: Bisection exited - 1000 iterations reached:\n")
        for i = 1:length(um)
            fprintf("u%i = %.3f  C%i = %.3f  du%i = %.3f\n",...
                    i, uave(i), i, Cave(i), i, du(i))
        end
        fprintf("\n")
    end

    u = uave;
    C = Cave;
    n = count;
end