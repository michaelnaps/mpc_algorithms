function [u] = id_qp(q, c, m, L, H, thd0)
    % ID-CLF-QP
    % Status: working
    
    nq = length(q)/2;
    u  = zeros(3,1);
    [~,M,E] = statespace(q, u, c, m, L);
    %% actuator map
    Be = eye(3);
   
    dq = [q(2);q(4);q(6)];
    %% virtual constraints
    %     [y_a, Jy_a, dJy_a, y_d, dy_d, ddy_d] = compute_outputs_joints(q,thd0);
    [y_a, Jy_a, dJy_a, y_d, dy_d, ddy_d] = compute_outputs(q,m,L,H);
    dim_y = length(y_a);
    
    %% inverse dynamics quadratic programming (ID-QP)
    dy_a = Jy_a*[q(2);q(4);q(6)];
    
    %% inverse dynamics quadratic programming (ID-QP)
%     DLfy = [dJy_a, Jy_a];
    
    kp = 100;
    kd = 20;
    mu = kp*(y_a - y_d) + kd*(dy_a - dy_d);
    
    
    A = Jy_a;
    b = dJy_a*dq - ddy_d + mu;
    
    Hmat = blkdiag(A'*A,0.0*ones(3));   % ddq'*A'*A*ddq + 0.0*u'*u
    bmat = [2*A'*b; zeros(3,1)];        % 2*A'*b*ddq
    Aeq = [M -Be];
    beq = [-E];
    options = optimoptions('quadprog','Algorithm','interior-point-convex','display','off');
    
    lb = [-inf(nq,1); -3000*ones(dim_y,1)];  
    ub = [inf(nq,1); 3000*ones(dim_y,1)];
    
    
    Aineq = [];
    bineq = [];
    
    X = quadprog(Hmat,bmat,Aineq,bineq,Aeq,beq,lb,ub,[],options);
    
    
    ddq_des = X(1:3);
    u = X(4:6);
%     disp(t)
    
    
end