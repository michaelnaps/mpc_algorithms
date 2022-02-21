function [T, q] = mpc_control(P, T, q0, um, c, m, L, Cq, eps)    
    %% MPC Controller
    dt = T(2) - T(1);
    q = NaN(length(T), length(q0));
    q(1,:) = q0;
    for i = 2:length(T)
        
        tic;
        [u, C, n] = pbs.bisection(P, dt, q(i-1,1:6)', q(i-1,7:9)', um, c, m, L, Cq, eps);
        t = toc;
        [~, qc] = ode45(@(t,q) statespace(q, u, c, m, L), 0:dt:P*dt, q(i-1,1:6));
        q(i,:) = [qc(2,:), u', C', n, t];
        
    end
end