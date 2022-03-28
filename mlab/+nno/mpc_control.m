function [T, q] = mpc_control(P, T, q0, um, c, m, L, Cq, thd, eps, height, push)
    %% Setup
    dt = T(2) - T(1);
    q = NaN(length(T), length(q0));
    q(1,:) = pend_push(q0, T(1), push);
    
    %% Simulation Loop
    for i = 2:length(T)
        % Adjust Cost Function if Applicable
        thd = pend_height(T(i), L, thd, height);
        
        % Calculate Optimal Input and Next State
        tic
        [u, C, n, brk] = nno.newtons(P, dt, q(i-1,1:6)', q(i-1,7:9)', um, c, m, L, Cq, thd, eps);
        t = toc;
        qc = modeuler(P, dt, q(i-1,1:6)', u, c, m, L, 'Main Simulation Loop');
        
        % Push Pendulum if Applicable
        qnew = pend_push(qc(2,:), T(i), push);
        
        % Add Values to State Matrix
        q(i,:) = [qnew, u', C, n, t, brk];

    end
    
end