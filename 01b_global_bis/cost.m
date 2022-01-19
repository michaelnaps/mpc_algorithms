function C = cost(q0, u)
    %% Global Variables
    global P dt Cq;
    
    %% State Calculation
    % calculate the state over the given prediction horizon
    % with constant input
    [~, qc] = ode45(@(t,q) statespace(q, u), 0:dt:P*dt, q0);
    
    %% Cost Calculation
    % sum of cost over the prediction horizon states
    C = zeros(size(u));
    for i = 1:P+1
        C = C + Cq(qc(i,:));
    end
end