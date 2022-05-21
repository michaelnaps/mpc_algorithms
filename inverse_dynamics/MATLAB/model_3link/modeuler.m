function [q] = modeuler(P, dt, q0, u, c, m, L, loc)
    %% check for minimum stepsize
    dt_min = 1e-3;
    if (dt > dt_min)
        adj = dt/dt_min;
    else
        adj = 1;
    end

    %% Initialize Arrays
    Nq = length(q0);
    Pm = adj*P;  dtm = dt/adj;
    q  = Inf(P+1, Nq);
    qm = Inf(Pm+1, Nq);

    %% Modified Euler Method
    q(1,:) = q0';
    qm(1,:) = q0';
    for i = 1:Pm
        dq1 = statespace(qm(i,:), u, c, m, L)';
        qeu = qm(i,:) + dq1*dtm;
        dq2 = statespace(qeu, u, c, m, L)';
        qm(i+1,:) = qm(i,:) + 1/2*(dq1 + dq2)*dtm;

        if (rem(i,adj) == 0)
            q(i/adj+1,:) = qm(i+1,:);
        end

        if (sum(isnan(qm(i+1,:))) > 0)
            fprintf("ERROR: statespace() returned NaN for inputs.\n")
            fprintf("location: %s\niteration(s): %i\n\n", loc, i)
            pause(1);
            break;
        end
    end
end

