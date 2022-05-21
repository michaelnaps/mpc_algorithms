%% Project: Angular Linear Inverted Pendulum Model
%  Created by: Michael Napoli

restoredefaultpath

clc;clear;
close all;

addpath ./model_3link

%% External Disturbance Testing
push = [0.75, 3, 2.0];
% push = [
%      0.50, 3,  5.0;
%      3.50, 3, -2.0;
%      3.75, 2,  3.0
%     ];

height = [0.00, 1];
% height = [
%      0.00, 1.6;
%      1.00, 1.2;
%      3.50, 1.5;
%      6.00, 1.9;
%      7.50, 1.6
%     ];
% h_t = (0:0.2:10)';  h = [linspace(1,2,5/0.2)';linspace(2,1,5/0.2+1)'];
% height = [h_t, h];

%% Mass, Length, Height and Angle Constants
% parameters for mass and length
m = [10; 10; 60]/2;
L = [0.4; 0.4; 0.6];
% calculate desired joint angles
thd0 = pend_angles(L, height(1,2));



%% Variable Setup
% establish state space vectors and variables
P = 1;                          % prediction horizon [time steps]
dt = 0.025;                     % change in time
T = 0:dt:3;                    % time span
th1_0 = [thd0(1); 0];            % link 1 position and velocity
th2_0 = [thd0(2); 0];            % link 2 position and velocity
th3_0 = [thd0(3); 0];            % link 3 position and velocity
um = [3000; 3000; 3000];        % maximum input to joints
c = 0*[500; 500; 500];            % damping coefficients
H_d = 0.8333; % desired height
% create initial states
q0 = [
      th1_0;th2_0;th3_0;...       % initial joint states
     ];

%% Simulation Loop
dt = T(2) - T(1);
q = NaN(length(T), 6);
u = NaN(length(T), 3);
q(1,:) = q0;
for i = 2:length(T)
%     uc = computed_torques(q(i-1,:)',c,m,L,h,thd0); % no QP
    uc = id_qp(q(i-1,:)',c,m,L,H_d,thd0); % inverse dynamics QP
%     uc = id_clf_qp(q(i-1,:)',c,m,L,h,thd0); % 
    % Adjust Cost Function if Applicable
    qc = modeuler(P, dt, q(i-1,:)', uc, c, m, L, 'Main Simulation Loop');

    % Push Pendulum if Applicable
    qnew = pend_push(qc(2,:), T(i), push);

    % Add Values to State Matrix
    q(i,:) = qnew;
    u(i,:) = uc';
    
    disp(uc');  disp(qnew);

end

%%
% velocity and position of link 1
jointStates_fig = figure('Position', [0 0 1400 800]);
hold on
subplot(2,3,1)
yyaxis left
plot(T, q(:,1))
ylabel('Pos [rad]')
yyaxis right
plot(T, q(:,2))
ylabel('Vel [rad/s]')
xlabel('Time [s]')
title('Link 1')
legend('Pos', 'Vel')

% velocity and position of link 2
subplot(2,3,2)
yyaxis left
plot(T, q(:,3))
ylabel('Pos [rad]')
yyaxis right
plot(T, q(:,4))
ylabel('Vel [rad/s]')
xlabel('Time [s]')
title('Link 2')
legend('Pos', 'Vel')

% velocity and position of link 3
subplot(2,3,3)
yyaxis left
plot(T, q(:,5))
ylabel('Pos [rad]')
yyaxis right
plot(T, q(:,6))
ylabel('Vel [rad/s]')
xlabel('Time [s]')
title('Link 3')
legend('Pos', 'Vel')

% plot input on link 1
subplot(2,3,4)
plot(T, u(:,1))
title('Input on Link 1')
ylabel('Input [Nm]')
xlabel('Time [s]')

% plot input on link 2
subplot(2,3,5)
plot(T, u(:,2))
title('Input on Link 2')
ylabel('Input [Nm]')
xlabel('Time [s]')
% hold off

% plot input on link 3
subplot(2,3,6)
plot(T, u(:,3))
title('Input on Link 3')
ylabel('Input [Nm]')
xlabel('Time [s]')
% hold off

%% export plots to png files
% exportgraphics(jointStates_fig, './nno_Figures/nno_singlePushRecovery.tif', 'resolution', 1200)
% exportgraphics(calcTime_fig, './nno_Figures/nno_calculationTime.tif', 'resolution', 1200)

% % animation of 3-link pendulum
% figure(3)
% T_anim = 0:0.05:T(end);
% q_anim = interp1(T,q,T_anim);
% animation_3link(q_anim, T_anim, m, L);

%%
% Step 1: implement ID-QP (or ID-CLF-QP) in python with 3link model
%      - (M) Control objectives: track 3 task-space outputs: (xcom, zcom, qtor)
%      - (A) Test controlling angular momentum (to be zero) with ID-QP or ID-CLF-QP
% Step 2: test angular momentum ID-QP in Python
% Step 3: Integrage with ALIP (in python)
%      - Ankle torque <= ALIP MPC
%      - Knee/Hip torques <= ID-QP (ID-CLF-QP)
%                            - control objectives: (zcom, desired angular momentum <- ALIP MPC))
%                            - auxilary objectives: qtor
%                            
% Step 4: transfer it to DIGIT
%      - how to convert ankle torque? 