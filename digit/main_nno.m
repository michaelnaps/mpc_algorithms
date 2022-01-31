%% Setting up path
restoredefaultpath; matlabrc;

clc;clear;
close all;

addpath /home/michaelnaps/prog/frost-dev
addpath /home/michaelnaps/prog/mpc_algorithms/digit

use_mathematica_sym = true;

if use_mathematica_sym
    load_path = 'gen/sym';
else
    load_path = [];
end

export_path = 'gen/mex';
utils.init_path(export_path, load_path);

frost_addpath;

%% load robot model
cur = utils.get_root_path();
urdf = fullfile(cur,'urdf','digit_model.urdf');
digit = sys.LoadModel(urdf, load_path);

%% Desired Values and Cost Function
qd = load('x0.mat');
% Generalized Cost Function
Cq = @(qd, q, du) 100*((cos(qd(1)) - cos(q(1)))^2 + (sin(qd(1)) - sin(q(1)))^2) + (qd(2) - q(2))^2 + 1e-7*(du)^2;

%% MPC Variable Setup
act_indices = find(cellfun(@(x)~isempty(x),{digit.Joints.Actuator}));
Params.P   = 4;
Params.dt  = 0.025;
Params.um  = 3000;
Params.Cq  = Cq;
Params.eps = 1e-6;
Params.qd = qd.q0(act_indices);
Params.dqd = qd.dq0(act_indices);
Params.a_ind = act_indices';

%% setup controller configurations
joint_torque = digit.Inputs.torque;
joint_torque.CallbackFunction = @nno.mpc_root;
joint_torque.Params = Params;
 
%% configure external input (disturbance)
torso_dist = InputVariable('f',3);
torso = sys.frames.torso(digit);
Jb_torso = getBodyJacobian(digit, torso);
setGmap(torso_dist, transpose(Jb_torso(1:3,:)), digit);

torso_dist.CallbackFunction = @sim.torso_disturbance;
digit.addInput(torso_dist);

%% Compile stuff if needed
compile = false;
if compile
    digit.compile(export_path);    
    load_path = 'gen/sym';
    if ~isfolder(load_path)
        mkdir(load_path);
    end
    saveExpression(digit, load_path); % run this after loaded the optimization problem
    
    export(torso_dist, export_path);
end

%% simulate
load('x0.mat');
x0  = [q0; dq0];
opts = struct;
[sol, logger] = simulate(digit, x0, 0, 1, {}, opts);
    
%%
plot.plotStates(digit, logger.flow);

%% Animate Ga
[conGUI] = plot.LoadSimAnimator(digit, logger.flow.t, logger.flow.states.x, 'SkipExporting', true, 'ExportPath',export_path);

