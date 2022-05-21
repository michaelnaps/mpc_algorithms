function [thd] = pend_angles(L, h)
    %% Length and Height Parameters for Link 1
    l1 = L(1);  l3 = L(3);
    h12 = (h - l3)/2;
    
    %% Calculate Necessary Angles
    thd = [
         asin(h12/l1);
         pi - 2*asin(h12/l1);
         1/2*(2*asin(h12/l1) - pi);
        ];
end