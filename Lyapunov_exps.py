import numpy as np
from scipy.integrate import solve_ivp

def calc_E(y):
    """Return the total energy of the system."""
    m1 = m2 = L1 = L2 = 1
    g = 9.81
    th1, th1d, th2, th2d = y
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
            2*L1*L2*th1d*th2d*np.cos(th1-th2))
    return T + V

def deriv(t, y, L1, L2, m1, m2, g):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def est_Lyapunov(x, eps, *, T=1, n=10000, args=[1, 1, 1, 1, 9.81]):
    l_exps = []
    for i in range(n):
        eps = 0.01/np.linalg.norm(eps)*eps    
        y = solve_ivp(deriv, (0, T), x, args=args, method='RK23')
        # if i % 100 == 0:
            # print(calc_E(x))

        y_dash = solve_ivp(deriv, (0, T), x+eps, args=args, method='RK23')
        x = np.array([y.y[0][-1], y.y[1][-1], y.y[2][-1], y.y[3][-1]])
        x_dash = np.array([y_dash.y[0][-1], y_dash.y[1][-1], y_dash.y[2][-1],
                           y_dash.y[3][-1]])
        eps = x_dash - x
        l_exp = np.log(np.linalg.norm(eps)/0.01)/T
        l_exps.append(l_exp)
    return sum(l_exps)/n  

eps = np.array([np.pi/3, 0, np.pi/3, 0])
x = np.array([0, 0, 0, 0])