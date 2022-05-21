function [u] = id_clf_qp(q, c, m, L, H, thd0)
    % ID-CLF-QP
    % Status: working
    nq = length(q)/2;
    u  = zeros(3,1);
    [~,M,E] = statespace(q, u, c, m, L);
    %% actuator map
    Be = eye(3);
   
    
    %% virtual constraints
    [y_a, Jy_a, dJy_a, y_d, dy_d, ddy_d] = compute_outputs(q,m,L,H);
    %     [y_a, Jy_a, dJy_a, y_d, dy_d, ddy_d] = compute_outputs_joints(q,thd0);
    dim_y = length(y_a);
    
    %% inverse dynamics quadratic programming (ID-QP)
    dy_a = Jy_a*[q(2);q(4);q(6)];

    ep = 0.1;
    F_mat = [zeros(dim_y), eye(dim_y);
        zeros(dim_y), zeros(dim_y)];
    G_mat = [zeros(dim_y);eye(dim_y)];
    
    weights = ones(1,dim_y);
    W = diag(weights);
    I_mat = blkdiag(1/ep*W, W);    
    eta = [y_a-y_d; dy_a-dy_d];
    
    % Lyapunov function
    Q = eye(dim_y*2);
    P = icare(F_mat,G_mat,Q);
    Pep = I_mat' * P * I_mat;
    Veta = eta' * Pep * eta;
    
    
    A_mat = Jy_a;
    b_vec = dJy_a*[q(2);q(4);q(6)] - ddy_d;% + mu;

    LgVetau = 2*eta'*Pep*G_mat*A_mat;
    gamma = (min(eig(Q))/max(eig(P)))/ep;  % 0.3660 = min(eig(Q)/max(eig(P));
    LfVetau = eta'*(F_mat'*Pep+Pep*F_mat)*eta + gamma*Veta +...
        2*eta'*Pep*G_mat*(b_vec);
    
    % ID-CLF-QP+-delta
    p = 1;
    Aineq = [LgVetau, zeros(1,dim_y), -p];
    bineq = -LfVetau;
    
    
    
    bmat = [2*A_mat'*b_vec + LgVetau'; zeros(dim_y,1); 0];
    %     bmat = [2*A_mat'*b_vec; zeros(dim_y,1); 0];
    Hmat = blkdiag(A_mat'*A_mat,0.001*ones(dim_y),p);
    
    
    Aeq = [M -Be  zeros(nq,1)];
    beq = -E;
        
    % vars: ddq, u, lambda, delta
    lb = [-3000*ones(nq,1); -[3000,3000,3000]'; -0];  
    ub = [3000*ones(nq,1); [3000,3000,3000]'; 0];
    
    
    
    
    options = optimoptions('quadprog','Algorithm','interior-point-convex','display','off');
    %     tic
    [X,fval,exitflag,output,~] = quadprog(Hmat,bmat,Aineq,bineq,Aeq,beq,lb,ub,[],options);
    %     toc
    %     disp(t)
    %     iter = output.iterations;
    
    if exitflag <= 0
        1
    end
    
    ddq_des = X(1:3);
    u = X(4:6);
    delta = X(7);
    
    
    
end