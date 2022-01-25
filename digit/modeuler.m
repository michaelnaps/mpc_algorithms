function [q] = modeuler(P, dt, q0, u, model)
    %% Initialize Arrays
    Pm = 10*P;  dtm = dt/10;
    q = Inf(P+1, length(q0));
    qm = Inf(Pm+1, length(q0));

    %% Modified Euler Method
    N = length(q0)/2;
    q(1,:) = q0';
    qm(1,:) = q0';
    for i = 1:Pm
        M = calcMassMatrix(model, qm(i,1:N));
        F = calcDriftVector(model, qm(i,1:N), qm(i,N:end));
        dq1 = M\(u - F);
        qeu = qm(i,:) + dq1*dtm;

        M = calcMassMatrix(model, qeu(1:N));
        F = calcDriftVector(model, que(1:N), qeu(N:end));
        dq2 = M\(u - F);
        qm(i+1,:) = qm(i,:) + 1/2*(dq1 + dq2)*dtm;

        if (rem(i,10) == 0)
            q(i/10+1,:) = qm(i+1,:);
        end

        if (sum(isnan(qm(i+1,:))) > 0)
            fprintf("ERROR: M/(u - F) returned NaN for inputs.\n")
            fprintf("iteration(s): %i\n\n", i)
            pause(1);
            break;
        end
    end
end

