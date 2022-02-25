function [u, n, brk, a] = alpha_bis(model, g, P, dt, q0, u0, uc, Cq, qd, arng, eps)
    %% Variable Setup
    if (length(arng) == 1)
        u = uc - arng*g;
        n = 0;
        brk = -2;
        a = arng;
        return;
    end

    alow = arng(1);
    ahgh = arng(2);
    aave = (ahgh + alow)/2;

    ulow = uc - alow*g;
    uhgh = uc - ahgh*g;
    uave = uc - aave*g;

    Clow = ngd.cost(model, P, dt, q0, u0, ulow, Cq, qd);
    Chgh = ngd.cost(model, P, dt, q0, u0, uhgh, Cq, qd);
    Cave = ngd.cost(model, P, dt, q0, u0, uave, Cq, qd);

    count = 1;
    brk = 0;
    while (Cave > eps)

        if (Clow < Chgh)
            ahgh = aave;
            Chgh = Cave;
        else
            alow = aave;
            Clow = Cave;
        end

        aave = (ahgh + alow)/2;
        uave = uc - aave*g;
        Cave = ngd.cost(model, P, dt, q0, u0, uave, Cq, qd);

        adiff = ahgh - alow;
        
        if (adiff < 1e-9)
            brk = 1;
            break;
        end
    
        count = count + 1;
        fprintf("\tIteration: %i, a: %.9f\n", count, aave)

        if (count == 1000)
            fprintf("ERROR: Iteration break in alpha search. (%i)\n", count)
            brk = -1;
            break;
        end
    end

    u = uave;
    a = aave;
    n = count;
end