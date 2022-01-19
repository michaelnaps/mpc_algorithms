function [c] = nlconstraints(qf)
    pmax = pi/2;
    wmax = 5;
    c = [
         abs(qf(1)) - pmax > 0;
         abs(qf(2)) - wmax > 0;
         abs(qf(3)) - pmax > 0;
         abs(qf(4)) - wmax > 0;
         abs(qf(5)) - pmax > 0;
         abs(qf(6)) - wmax > 0
        ];
end