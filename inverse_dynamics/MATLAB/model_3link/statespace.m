function [dq,M,E] = statespace(q, u, c, m, L)
    %% Setup
    m1 = m(1);      m2 = m(2);      m3 = m(3);
    L1 = L(1);      L2 = L(2);      L3 = L(3);
    r1 = L1/2;      r2 = L2/2;      r3 = L3/2;
    I1 = m1*L1/12;  I2 = m2*L2/12;  I3 = m3*L3/12;
    g = 9.81;
    
    q1 = q(1);  q2 = q(3);  q3 = q(5);
    q4 = q(2);  q5 = q(4);  q6 = q(6);
    
    u1 = u(1); u2 = u(2); u3 = u(3);
    c1 = c(1); c2 = c(2); c3 = c(3);

    %% State Space Equations
    % Equation: E*ddq = M (rearrange for ddq)
    M(1,1) = -I3 - m3*r3^2 - L1*m3*r3*cos(q2 + q3) - L2*m3*r3*cos(q3);
    M(1,2) = -m3*r3^2 - L2*m3*cos(q3)*r3 - I3;
    M(1,3) = -m3*r3^2 - I3;
    M(2,1) = -m3*L2^2 - 2*m3*cos(q3)*L2*r3 - L1*m3*cos(q2)*L2 - m2*r2^2 - L1*m2*cos(q2)*r2 - m3*r3^2 - L1*m3*cos(q2 + q3)*r3 - I2 - I3;
    M(2,2) = -m3*L2^2 - 2*m3*cos(q3)*L2*r3 - m2*r2^2 - m3*r3^2 - I2 - I3;
    M(2,3) = -m3*r3^2 - L2*m3*cos(q3)*r3 - I3;
    M(3,1) = -I1 - I2 - I3 - L1^2*m2 - L1^2*m3 - L2^2*m3 - m1*r1^2 - m2*r2^2 - m3*r3^2 - 2*L1*m3*r3*cos(q2 + q3) - 2*L1*L2*m3*cos(q2) - 2*L1*m2*r2*cos(q2) - 2*L2*m3*r3*cos(q3);
    M(3,2) = -m3*L2^2 - 2*m3*cos(q3)*L2*r3 - L1*m3*cos(q2)*L2 - m2*r2^2 - L1*m2*cos(q2)*r2 - m3*r3^2 - L1*m3*cos(q2 + q3)*r3 - I2 - I3;
    M(3,3) = -I3 - m3*r3^2 - L1*m3*r3*cos(q2 + q3) - L2*m3*r3*cos(q3);

    E = [
         g*m3*r3*cos(q1 + q2 + q3) + c3*L3*q6 - u3 + L1*m3*r3*q4^2*sin(q2 + q3) + L2*m3*r3*q4^2*sin(q3) + L2*m3*r3*q5^2*sin(q3) + 2*L2*m3*r3*q4*q5*sin(q3);
         L2*g*m3*cos(q1 + q2) + c2*L2*q5 - u2 + g*m2*r2*cos(q1 + q2) + g*m3*r3*cos(q1 + q2 + q3) + L1*m3*r3*q4^2*sin(q2 + q3) + L1*L2*m3*q4^2*sin(q2) + L1*m2*r2*q4^2*sin(q2) - L2*m3*r3*q6^2*sin(q3) - 2*L2*m3*r3*q4*q6*sin(q3) - 2*L2*m3*r3*q5*q6*sin(q3);
         L2*g*m3*cos(q1 + q2) + c1*L1*q4 - u1 + g*m2*r2*cos(q1 + q2) + L1*g*m2*cos(q1) + L1*g*m3*cos(q1) + g*m1*r1*cos(q1) + g*m3*r3*cos(q1 + q2 + q3) - L1*m3*r3*q5^2*sin(q2 + q3) - L1*m3*r3*q6^2*sin(q2 + q3) - L1*L2*m3*q5^2*sin(q2) - L1*m2*r2*q5^2*sin(q2) - L2*m3*r3*q6^2*sin(q3) - 2*L1*m3*r3*q4*q5*sin(q2 + q3) - 2*L1*m3*r3*q4*q6*sin(q2 + q3) - 2*L1*m3*r3*q5*q6*sin(q2 + q3) - 2*L1*L2*m3*q4*q5*sin(q2) - 2*L1*m2*r2*q4*q5*sin(q2) - 2*L2*m3*r3*q4*q6*sin(q3) - 2*L2*m3*r3*q5*q6*sin(q3)
        ];

    M = -flip(M);
    E = flip(E);

    ddq = M\(-E);

    dq = [q(2); ddq(1); q(4); ddq(2); q(6); ddq(3)];
end