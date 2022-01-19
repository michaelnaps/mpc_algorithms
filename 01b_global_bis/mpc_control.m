function [T, q] = mpc_control(T, q0)
    %% Global Variables
    global P dt;

    %% MPC Controller
    q = NaN(length(T), length(q0));
    q(1,:) = q0;
    for i = 2:length(T)
        
        [u, C, n] = bisection(q(i-1,1:6));
        [~, qc] = ode45(@(t,q) statespace(q, u), 0:dt:P*dt, q(i-1,1:6));
        q(i,:) = [qc(2,:), u', C', n];
        
    end
end