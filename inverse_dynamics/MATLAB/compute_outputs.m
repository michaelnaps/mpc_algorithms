function [y_a, Jy_a, dJy_a, y_d, dy_d, ddy_d] = compute_outputs(q,m,L,H)
% compute actual and desired outputs
% outputs to be controlled
% - x_com
% - z_com
% - q_tor

%% create current state variables (y_as)
% location and angle of CoM
th1  = q(1);     th2  = q(3);    th3  = q(5);
dth1 = q(2);     dth2 = q(4);    dth3 = q(6);
m1   = m(1);     m2   = m(2);    m3   = m(3);
L1   = L(1);     L2   = L(2);    L3   = L(3);
r1   = L1/2;     r2   = L2/2;    r3   = L3/2;

xAnkle = 0; yAnkle = 0;

x_Link1_CoM = xAnkle + r1*cos(th1);
y_Link1_CoM = yAnkle + r1*sin(th1);

xKnee = xAnkle + L1*cos(th1);
yKnee = yAnkle + L1*sin(th1);

x_Link2_CoM = xKnee + r2*cos(th1+th2);
y_Link2_CoM = yKnee + r2*sin(th1+th2);

xHip = xKnee + L2*cos(th1+th2);
yHip = yKnee + L2*sin(th1+th2);

x_Link3_CoM = xHip + r3*cos(th1+th2+th3);
y_Link3_CoM = yHip + r3*sin(th1+th2+th3);

x_com = (m1*x_Link1_CoM + m2*x_Link2_CoM + m3*x_Link3_CoM) / (m1 + m2 + m3);
z_com = (m1*y_Link1_CoM + m2*y_Link2_CoM + m3*y_Link3_CoM) / (m1 + m2 + m3);

q_tor = th1 + th2 + th3;

y_a = [x_com
    z_com
    q_tor];

%%
% Jya = jacobian(ya, [th1;th2;th3]);
Jac_x_com = [-(m3*(r3*sin(th1 + th2 + th3) + L2*sin(th1 + th2) + L1*sin(th1)) + m2*(r2*sin(th1 + th2) + L1*sin(th1)) + m1*r1*sin(th1))/(m1 + m2 + m3),...
    -(m3*(r3*sin(th1 + th2 + th3) + L2*sin(th1 + th2)) + m2*r2*sin(th1 + th2))/(m1 + m2 + m3),...
    -(m3*r3*sin(th1 + th2 + th3))/(m1 + m2 + m3)];

Jac_z_com = [(m2*(r2*cos(th1 + th2) + L1*cos(th1)) + m3*(L2*cos(th1 + th2) + L1*cos(th1) + r3*cos(th1 + th2 + th3)) + m1*r1*cos(th1))/(m1 + m2 + m3),...
    (m3*(L2*cos(th1 + th2) + r3*cos(th1 + th2 + th3)) + m2*r2*cos(th1 + th2))/(m1 + m2 + m3),...
    (m3*r3*cos(th1 + th2 + th3))/(m1 + m2 + m3)];

Jac_q_tor = [1, 1, 1];

Jy_a = [Jac_x_com
    Jac_z_com
    Jac_q_tor];

% dy_a = Jy_a*[dth1;dth2;dth3];
% dJy_a = jacobian(dy_a, [th1;th2;th3]);
d_Jac_x_com = [- (dth1*(m2*(r2*cos(th1 + th2) + L1*cos(th1)) + m3*(L2*cos(th1 + th2) + L1*cos(th1) + r3*cos(th1 + th2 + th3)) + m1*r1*cos(th1)))/(m1 + m2 + m3) - (dth2*(m3*(L2*cos(th1 + th2) + r3*cos(th1 + th2 + th3)) + m2*r2*cos(th1 + th2)))/(m1 + m2 + m3) - (dth3*m3*r3*cos(th1 + th2 + th3))/(m1 + m2 + m3),...
    - (dth1*(m3*(L2*cos(th1 + th2) + r3*cos(th1 + th2 + th3)) + m2*r2*cos(th1 + th2)))/(m1 + m2 + m3) - (dth2*(m3*(L2*cos(th1 + th2) + r3*cos(th1 + th2 + th3)) + m2*r2*cos(th1 + th2)))/(m1 + m2 + m3) - (dth3*m3*r3*cos(th1 + th2 + th3))/(m1 + m2 + m3),...
    - (dth1*m3*r3*cos(th1 + th2 + th3))/(m1 + m2 + m3) - (dth2*m3*r3*cos(th1 + th2 + th3))/(m1 + m2 + m3) - (dth3*m3*r3*cos(th1 + th2 + th3))/(m1 + m2 + m3)];

d_Jac_z_com = [- (dth2*(m3*(r3*sin(th1 + th2 + th3) + L2*sin(th1 + th2)) + m2*r2*sin(th1 + th2)))/(m1 + m2 + m3) - (dth1*(m3*(r3*sin(th1 + th2 + th3) + L2*sin(th1 + th2) + L1*sin(th1)) + m2*(r2*sin(th1 + th2) + L1*sin(th1)) + m1*r1*sin(th1)))/(m1 + m2 + m3) - (dth3*m3*r3*sin(th1 + th2 + th3))/(m1 + m2 + m3),...
    - (dth1*(m3*(r3*sin(th1 + th2 + th3) + L2*sin(th1 + th2)) + m2*r2*sin(th1 + th2)))/(m1 + m2 + m3) - (dth2*(m3*(r3*sin(th1 + th2 + th3) + L2*sin(th1 + th2)) + m2*r2*sin(th1 + th2)))/(m1 + m2 + m3) - (dth3*m3*r3*sin(th1 + th2 + th3))/(m1 + m2 + m3),...
    - (dth1*m3*r3*sin(th1 + th2 + th3))/(m1 + m2 + m3) - (dth2*m3*r3*sin(th1 + th2 + th3))/(m1 + m2 + m3) - (dth3*m3*r3*sin(th1 + th2 + th3))/(m1 + m2 + m3)];

d_Jac_q_tor = [0, 0, 0];

dJy_a = [d_Jac_x_com
    d_Jac_z_com
    d_Jac_q_tor];
% ddy_a = dJy_a*[dth1;dth2;dth3] + Jy_a*[ddth1;ddth2;ddth3];


% constant desired outputs
y_d = [0
    H
    pi/2]; 

dy_d = zeros(3,1);
ddy_d = zeros(3,1);



end