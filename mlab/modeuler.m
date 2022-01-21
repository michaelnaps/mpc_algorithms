function [q] = modeuler(P, dt, q0, u, c, m, L, loc)
    %% Initialize Arrays
    Pm = 10*P;  dtm = dt/10;
    q = Inf(P+1, length(q0));
    qm = Inf(Pm+1, length(q0));

    %% Modified Euler Method
    q(1,:) = q0';
    qm(1,:) = q0';
    for i = 1:Pm
        dq1 = statespace(qm(i,:), u, c, m, L)';
        qeu = qm(i,:) + dq1*dtm;
        dq2 = statespace(qeu, u, c, m, L)';
        qm(i+1,:) = qm(i,:) + 1/2*(dq1 + dq2)*dtm;

        if (rem(i,10) == 0)
            q(i/10+1,:) = qm(i+1,:);
        end

        if (sum(isnan(qm(i+1,:))) > 0)
            fprintf("ERROR: statespace() returned NaN for inputs.\n")
            fprintf("location: %s\niteration(s): %i\n\n", loc, i)
            pause(1);
            break;
        end
    end
end

