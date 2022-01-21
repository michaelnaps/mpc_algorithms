function Ct = cost(P, dt, q0, u, c, m, L, Cq)
    %% Cost of Constant Input
    % calculate the state over the desired prediction horizon
    qc = modeuler(P, dt, q0, u, c, m, L);
    
    % sum of cost of each step of the prediction horizon
    C = zeros(size(u'));
    for i = 1:P+1
        C = C + Cq(qc(i,:));
    end
    Ct = sum(C);
end