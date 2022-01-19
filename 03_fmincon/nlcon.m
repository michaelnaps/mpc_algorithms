function [c, ceq] = nlcon(P, dt, q0, u, c, m, L)
    pmax = pi/2;
    wmax = 5;

    qn = modeuler(P, dt, q0, u, c, m, L);

    c = [
         abs(qn(end,1)) - pmax;
         abs(qn(end,2)) - wmax;
         abs(qn(end,3)) - pmax;
         abs(qn(end,4)) - wmax;
         abs(qn(end,5)) - pmax;
         abs(qn(end,6)) - wmax
        ];
    ceq = [];
end