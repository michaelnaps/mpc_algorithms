function [u, n, brk, a] = alpha_blk(g, Cc, P, dt, q0, u0, uc, c, m, L, Cq, qd, arng, eps)
    tao = 0.90;
    ablk = arng(2);
    ublk = uc - ablk*g;
    Cblk = ngd.cost(P, dt, q0, u0, ublk, c, m, L, Cq, qd, " Alpha Search Function - Initial ");

    count = 1;
    brk = 0;
    while (Cblk > Cc)
        ablk = tao*ablk;
        ublk = uc - ablk*g;
        Cblk = ngd.cost(P, dt, q0, u0, ublk, c, m, L, Cq, qd, " Alpha Search Function - Initial ");
        count = count + 1;

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