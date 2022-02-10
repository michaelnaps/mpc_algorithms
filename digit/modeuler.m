function [tspan, q] = modeuler(odefun, tspan, q0, ~)
    %% Initialize Arrays/Matrices
    adj = 10;
    
    P = length(tspan) - 1;
    Pm = adj*P;
    
    dt = tspan(2) - tspan(1);
    dtm = dt/adj;
    tm = tspan(1):dt/adj:tspan(end);

    q = Inf(P+1, length(q0));
    qm = Inf(Pm+1, length(q0));

    %% Modified Euler Method
    q(1,:) = q0';
    qm(1,:) = q0';
    for i = 1:Pm
        dq1 = odefun(tm(i), qm(i,:)');
        qeu = qm(i,:)' + dq1*dtm;
        dq2 = odefun(tm(i), qeu);
        qm(i+1,:) = (qm(i,:)' + 1/2*(dq1 + dq2)*dtm)';

        if (rem(i,adj) == 0)
            q(i/adj+1,:) = qm(i+1,:);
        end

        if (sum(isnan(qm(i+1,:))) > 0)
            fprintf("ERROR: odefun() returned NaN for inputs.\n")
            fprintf("iteration(s): %i\n\n", i)
            break;
        end
    end
end

