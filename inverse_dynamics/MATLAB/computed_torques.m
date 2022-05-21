function [u] = computed_torques(q, c, m, L, H, thd0)
    % input-output linearization control
    
    u  = zeros(3,1);
    [~,M,E] = statespace(q, u, c, m, L);
    
    dq = [q(2);q(4);q(6)];
    %% actuator map
    Be = eye(3);    
    % compute vector fields
    % f(x)
    vfc = [
        dq;
        M \ (-E)];
    
    
    % g(x)
    gfc = [
        zeros(size(Be));
        M \ Be];
    
    
    %     p_sw_des
    %% virtual constraints
    %     [y_a, Jy_a, dJy_a, y_d, dy_d, ddy_d] = compute_outputs_joints(q,thd0);
    [y_a, Jy_a, dJy_a, y_d, dy_d, ddy_d] = compute_outputs(q,m,L,H);
    dy_a = Jy_a*[q(2);q(4);q(6)];
    DLfy = [dJy_a, Jy_a];
    
    kp = 100;
    kd = 20;
    mu = kp*(y_a - y_d) + kd*(dy_a - dy_d);
    
    
    
    % decoupling matrix
    A_mat  = DLfy*gfc;
    % feedforward term
    Lf_mat = DLfy*vfc;
    u_ff = - A_mat \ (Lf_mat - ddy_d);
    u_fb = - A_mat \ mu;
            
    u = u_ff + u_fb;
    
    
   
end

