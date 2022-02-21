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
function [u, C, n] = bisection(P, dt, q0, u0, um, c, m, L, Cq, eps)
    %% Initial Cost Check
    C0 = pbs.cost(P, dt, q0, u0, u0, c, m, L, Cq);
    if (sum(C0 > eps) < 1)
        u = u0;
        C = C0;
        n = 0;
        return;
    end

    %% Setup
    N = length(u0);
    ua = -um;
    ub =  um;
    uave = zeros(size(u0));
    du = Inf(size(u0));
    Ca = pbs.cost(P, dt, q0, u0, ua, c, m, L, Cq);
    Cb = pbs.cost(P, dt, q0, u0, ub, c, m, L, Cq);
    Cave = pbs.cost(P, dt, q0, u0, uave, c, m, L, Cq);
    
    %% Optimization Loop
    count = 1;
    while (sum(Cave > eps) > 0)
        
        % update boundary variables
        for i = 1:N
            
            if (du(i) < eps)
                break;
            end

            if (Ca(i) < Cb(i))
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
        if (sum(du < eps) == N)
            break;
        end
        
        % update center inputs and cost
        uave = (ua + ub) ./ 2;
        Cave = pbs.cost(P, dt, q0, u0, uave, c, m, L, Cq);
        count = count + 1;
        
        % iteration check
        if (count == 1000)
            break;
        end

    end
    
    %% Post-Loop Processes/Checks
    if (count == 1000)
        fprintf("ERROR: Bisection exited - 1000 iterations reached:\n")
        for i = 1:N
            fprintf("u%i = %.3f  C%i = %.3f  du%i = %.3f\n",...
                    i, uave(i), i, Cave(i), i, du(i))
        end
        fprintf("\n")
    end

    u = uave;
    C = Cave;
    n = count;
end