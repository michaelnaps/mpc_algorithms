function [tspan, q, logger] = modeuler(model, odefun, tspan, q0, logger)
    %% Check that time step is above minimum (1e-3)
    minStepSize = 1e-3;
    dt = tspan(2) - tspan(1);
    if (dt > minStepSize)
        adj = dt/minStepSize;
    else
        adj = 1;
    end

    %% Initialize Arrays/Matrices
    P = length(tspan) - 1;
    Pm = adj*P;
    dtm = dt/adj;
    tm = tspan(1):dt/adj:tspan(end);
    q = Inf(P+1, length(q0));
    qm = Inf(Pm+1, length(q0));

    %% Modified Euler Method
    q(1,:) = q0';
    qm(1,:) = q0';
    for i = 1:Pm
        dq1 = odefun(model, tm(i), qm(i,:)', logger);
        qeu = qm(i,:)' + dq1*dtm;
        dq2 = odefun(model, tm(i+1), qeu, logger);
        qm(i+1,:) = (qm(i,:)' + 1/2*(dq1 + dq2)*dtm)';

        if (rem(i,adj) == 0)
            q(i/adj+1,:) = qm(i+1,:);
            updateLog(logger);
        end

        if (sum(isnan(qm(i+1,:))) > 0)
            fprintf("ERROR: in modeuler() -> odefun() returned NaN for inputs.\n")
            fprintf("iteration(s): %i\n\n", i)
            break;
        end
    end
end

