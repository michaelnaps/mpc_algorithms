function [T, q] = mpc_control(P, T, q0, um, c, m, L, Cq)
    %% MPC Controller
    options = optimoptions('fmincon','Display','off');
    dt = T(2) - T(1);
    q = NaN(length(T), length(q0));
    q(1,:) = q0';
    for i = 2:length(T)
        
        tic;
        q_opt = q(i-1,1:6)';
        [u, C, ~, o] = fmincon(@(u) cost(P,dt,q_opt,u,c,m,L,Cq),q(i-1,7:9),[],[],[],[],-um,um,@(u) nlcon(P,dt,q_opt,u,c,m,L),options);
        t = toc;
        [~, qc] = ode45(@(t,q) statespace(q,u,c,m,L), 0:dt:P*dt, q(i-1,1:6));
        q(i,:) = [qc(2,:), u, C, o.iterations, t];
        
    end
end