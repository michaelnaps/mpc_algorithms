function Ct = cost(P, dt, q0, u0, c, m, L, Cq)
    %% Cost of Constant Input
    % calculate the state over the desired prediction horizon
%     [~, qc] = ode45(@(t,q) statespace(q, u, c, m, L), 0:dt:P*dt, q0);

    uc = reshape(u0, [3 4]);
    
    qc = zeros(P+1, length(q0));
    qc(1,:) = q0;
    for i = 2:P+1
        qtemp = modeuler(10*P, dt/10, qc(i-1,:), uc(:,i-1), c, m, L);
        qc(i,:) = qtemp(10*P+1,:);
    end
    
    % sum of cost of each step of the prediction horizon
    C = zeros(size(L));
%     for i = 1:P+1
    for i = 1:P+1
        C = C + Cq(qc(i,:));
    end
    Ct = sum(C);
end