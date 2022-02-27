function [u, n, brk, a] = alpha_blk(model, g, Cc, P, dt, q0, u0, uc, Cq, qd, arng)
    tau = 0.5;  % step shrink-coefficient
    ablk = arng(2);
    ublk = uc - ablk*g;
    Cblk = ngd.cost(model, P, dt, q0, u0, ublk, Cq, qd);

    count = 1;
    brk = 0;
    while (Cblk > Cc)
        ablk = tau*ablk;
        ublk = uc - ablk*g;
        Cblk = ngd.cost(model, P, dt, q0, u0, ublk, Cq, qd);

        count = count + 1;
        fprintf("\tIteration: %i, a: %.3e\n", count, ablk)

        if (count == 1000)
            fprintf("ERROR: Iteration break in alpha search. (%i)\n", count)
            brk = -1;
            break;
        end
    end

    u = ublk;
    n = count;
    a = ablk;
end