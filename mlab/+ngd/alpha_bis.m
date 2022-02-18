function [u, n, brk, a] = alpha_bis(g, P, dt, q0, u0, uc, c, m, L, Cq, qd, arng, eps)
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

    Clow = ngd.cost(P, dt, q0, u0, ulow, c, m, L, Cq, qd, " Alpha Search Function - Low  1 ");
    Chgh = ngd.cost(P, dt, q0, u0, uhgh, c, m, L, Cq, qd, " Alpha Search Function - High 1 ");
    Cave = ngd.cost(P, dt, q0, u0, uave, c, m, L, Cq, qd, " Alpha Search Function - Ave  1 ");

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
        Cave = ngd.cost(P, dt, q0, u0, uave, c, m, L, Cq, qd, " Alpha Search Function - Loop ");

        adiff = ahgh - alow;
        
        if (adiff < eps)
            brk = 1;
            break;
        end

        count = count + 1;

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