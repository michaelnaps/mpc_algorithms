function [u] = id_qp(input, robot, t, x, logger)
    % ID-CLF-QP
    % Status: working
    
    nx = robot.Dimension;
    q = x(1:nx);
    dq = x(nx+1:end);    
    params = input.Params;

    %     u  = zeros(3,1);

    M = calcMassMatrix(robot, q);
    H = calcDriftVector(robot, q, dq);   
    %% actuator map
    B = eye(3);
   
    %% virtual constraints
    %     [y_a, Jy_a, dJy_a, y_d, dy_d, ddy_d] = compute_outputs_joints(q,thd0);
    [y_a, Jy_a, dJy_a, y_d, dy_d, ddy_d] = compute_outputs(q,dq,params);
    %     dim_y = length(y_a);
    
    %% inverse dynamics quadratic programming (ID-QP)
    dy_a = Jy_a*dq;
    
    %% inverse dynamics quadratic programming (ID-QP)
    %     DLfy = [dJy_a, Jy_a];
    
    kp = diag([200,50,100]);
    kd = diag([20,20,0]);
    mu = kp*(y_a - y_d) + kd*(dy_a - dy_d);
    
    L_c = centroidal_momentum(q,dq);
    JL_c = J_centroidal_momentum(q);
    dJL_c = dJ_centroidal_momentum(q,dq);
    
    % run MPC here
    
    % desired centroidal momentum
    L_c_des = 0;
    
    A = [Jy_a                % outputs
        JL_c'];              % momentum
    b = [dJy_a*dq - ddy_d + mu;  % outputs
        dJL_c*dq + 20*(L_c - L_c_des)]; % momentum
    
    %     A = Jy_a;
    %     b = dJy_a*dq - ddy_d + mu;


    Hmat = blkdiag(A'*A,0.0*ones(3));   % ddq'*A'*A*ddq + 0.0*u'*u
    bmat = [2*A'*b; zeros(3,1)];        % 2*A'*b*ddq
    Aeq = [M -B];
    beq = [-H];
    options = optimoptions('quadprog','Algorithm','interior-point-convex','display','off');
    
    lb = [-2000*ones(nx,1); -[40;1000;500]]; %ddq, u 
    ub = [2000*ones(nx,1); [40;1000;500]];
    
    
    % CBF will enforce the controller operates only in the safe region if
    % feasible
    enable_cbf = true;
    if enable_cbf
        gamma1 = 2;
        gamma2 = 2;

        h_con = -[y_a(1) + 0.1          % xcom < 0.1
            - y_a(1) + 0.1              % xcom > -0.1
            y_a(2) - 0.5                % ycom > 0.5
            -y_a(2) + 1                 % ycom < 1
            (y_a(3) - pi/2) + 1         % qtor - pi/2 < 1
            -(y_a(3) - pi/2) + 1];      % qtor - pi/2 > -1
        J_con = -[Jy_a(1,:);
            -Jy_a(1,:)
            Jy_a(2,:)
            -Jy_a(2,:)
            Jy_a(3,:)
            -Jy_a(3,:)];
        dJ_con = -[dJy_a(1,:);
            -dJy_a(1,:)
            dJy_a(2,:)
            -dJy_a(2,:)
            dJy_a(3,:)
            -dJy_a(3,:)];

        Aineq = [J_con,zeros(length(h_con),3)];
        bineq = -(dJ_con + gamma1*J_con)*dq - gamma2*(J_con*dq + gamma1*h_con);
    else
        Aineq = [];
        bineq = [];
    end
    
    X = quadprog(Hmat,bmat,Aineq,bineq,Aeq,beq,lb,ub,[],options);
    
    
    ddq_des = X(1:3);
    u = X(4:6);
    disp(t)
    
    calc = logger.calc;
    calc.ya = y_a;
    calc.yd = y_d;
    calc.dya = dy_a;
    calc.dyd = dy_d;
    calc.u = u;
    calc.mu = mu;
    calc.ddq_des = ddq_des;
    calc.L_c = L_c;
    logger.calc = calc;
end