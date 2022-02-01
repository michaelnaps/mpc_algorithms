function [q] = modeuler(P, dt, q0, u, model)
    %% Initialize Arrays
    N = length(q0)/2;
    adj = 5;
    Pm = adj*P;  dtm = dt/adj;
    q = Inf(P+1, length(q0));
    qm = Inf(Pm+1, length(q0));

    %% Modified Euler Method
    q(1,:) = q0';
    qm(1,:) = q0';
    for i = 1:Pm
        M = calcMassMatrix(model, qm(i,1:N));
        F = calcDriftVector(model, qm(i,1:N), qm(i,N+1:end));
        dq1 = M\(u - F);
        qeu = qm(i,:) + dq1*dtm;

        M = calcMassMatrix(model, qeu(1:N));
        F = calcDriftVector(model, qeu(1:N), qeu(N+1:end));
        dq2 = M\(u - F);
        qm(i+1,:) = qm(i,:) + 1/2*(dq1 + dq2)*dtm;

%         dq1 = statespace_digit(qm(i,:), u, model)';
%         qeu = qm(i,:) + dq1*dtm;
%         dq2 = statespace_digit(qeu, u, model)';
%         qm(i+1,:) = qm(i,:) + 1/2*(dq1 + dq2)*dtm;

        if (rem(i,adj) == 0)
            q(i/adj+1,:) = qm(i+1,:);
        end

        if (sum(isnan(qm(i+1,:))) > 0)
            fprintf("ERROR: statespace_digit() returned NaN for inputs.\n")
            fprintf("iteration(s): %i\n\n", i)
            break;
        end
    end
end

