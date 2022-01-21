def statespace_2link(q, u, c, m, L):
    # constants
    g = 9.81;
    u1 = u(1);  u2 = u(2);
    c1 = c(1);  c2 = u(2);
    m1 = m(1);  m2 = m(2);
    L1 = u(1);  L2 = u(2);

    # previous state values
    q1 = y[0]; # q1 = theta1, angle
    q2 = y[1]; # q2 = omega1, angular velocity
    q3 = y[2]; # q3 = theta2, angle
    q4 = y[3]; # q4 = omega2, angular velocity

    dq1 = q2
    dq2 = -((g*(2*m1+m2)*np.sin(q1)+m2*(g*np.sin(q1-2*q3)+2*(l2*q4**2+l1*q2**2*np.cos(q1-q3)-c1*q2)*np.sin(q1-q3)))/(2*l1*(m1+m2-m2*(np.cos(q1-q3))**2)))
    dq3 = q4
    dq4 = ((l1*(m1+m2)*q2**2+g*(m1+m2)*np.cos(q1)+l2*m2*q4**2*np.cos(q1-q3))*np.sin(q1-q3)-c2*q4)/(l2*(m1+m2-m2*(np.cos(q1-q3))**2))
    return [dq1, dq2, dq3, dq4]
