function [y_a, Jy_a, dJy_a, y_d, dy_d, ddy_d] = compute_outputs_joints(q,thd0)
% compute actual and desired outputs
% outputs to be controlled
% - x_com
% - z_com
% - q_tor

y_a = [q(1)
    q(3)
    q(5)];

Jy_a = eye(3);



dJy_a = zeros(3);
% ddy_a = dJy_a*[dth1;dth2;dth3] + Jy_a*[ddth1;ddth2;ddth3];


% constant desired outputs
y_d = thd0; 

dy_d = zeros(3,1);
ddy_d = zeros(3,1);



end