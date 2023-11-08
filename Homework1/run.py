import numpy as np
from matplotlib import pyplot as plt
from doc.stdatm import stdatm
from doc.naca16_509_m06_clcd import naca16_509_m06

# ------ SOLVER CLASS ------
class BEM_Solver:
    def __init__(self, N_blades, R, R_hub, c, beta, Coeff, N_elem, tol, twist:float = 0.) -> None:
        self.J = None
        self.N_blades = N_blades
        self.R = R
        self.R_hub = R_hub
        self.h = (R - R_hub)/N_elem
        self.r = np.linspace(R_hub + self.h/2, R - self.h/2, endpoint=True, num=N_elem)

        self.c = c(self.r)
        self.beta = beta(self.r) + np.radians(twist)
        self.Coeff = Coeff
        self.N_elem = N_elem
        self.tol = tol

        self.a = np.zeros(N_elem)
        self.a_prime = np.zeros(N_elem)

        self.solved = False

    def solve(self, omega):
        if self.J is None: raise(AssertionError("You should set a value for J"))
        a_k = 0
        a_prime_k = 0
        for i in range(self.N_elem):
            a_k_1 = a_k + 10 * self.tol
            a_prime_k_1 = a_prime_k + 10 * self.tol
            sigma = self.N_blades * self.c[i]/(2 * np.pi * self.r[i]) 
            while np.abs(a_k - a_k_1) >= self.tol and np.abs(a_prime_k - a_prime_k_1) >= self.tol:
                phi = np.arctan2(self.J * self.R/(np.pi * self.r[i]) * (1 + a_k), (1 - a_prime_k))
                alpha = self.beta[i] - phi
                C_L, C_D = self.Coeff(alpha)
                c_N = C_L * np.cos(phi) - C_D * np.sin(phi)
                c_T = C_L * np.sin(phi) + C_D * np.cos(phi)
                a_k_1 = a_k
                a_prime_k_1 = a_prime_k
                a_k = (1 - omega) * a_k_1 + omega * sigma * c_N * (1 + a_k_1)/(2 * (1 - np.cos(2 * phi)))
                a_prime_k = (1 - omega) * a_prime_k_1 + omega * sigma * c_T * (1 - a_prime_k_1)/(2 * np.sin(2 * phi))
            
            self.a[i] = a_k
            self.a_prime[i] = a_prime_k
        self.solved = True

    def get_coeff(self):
        if not self.solved:
            raise(AssertionError("The solve function should be called first"))
        
        k_T = 0
        k_Q = 0
        for i in range(self.N_elem):
            k_T += self.a[i] * (1 + self.a[i]) * self.r[i] * self.h
            k_Q += self.a_prime[i] * (1 + self.a[i]) * self.r[i]**3 * self.h

        k_T *= self.J*self.J * np.pi/(self.R * self.R)
        k_Q *= self.J * np.pi * np.pi/(2 * self.R**4)
        k_P = 2 * np.pi * k_Q
        eta = self.J * k_T/k_P

        return (k_T, k_Q, k_P, eta)


# ------ SOLVER PARAM ------
tol = 1e-8
N_elem = 50

# ------ DATA FLAT PLATE ------
R = 1/2
R_hub = 0.25/2
N_blades = 2
c = lambda r : 0.15 * np.ones_like(r)
beta = lambda r : np.radians(25) * np.ones_like(r)
Coeff = lambda a : (2 * np.pi * a, 0)

s1 = BEM_Solver(N_blades, R, R_hub, c, beta, Coeff, N_elem, tol)


# ------ DATA HAMILTON-STANDARD 24D50 - 6813 ------
R = 3.4/2
R_hub = 0.45/2
N_blades = 4
c = lambda r : 0.15 * np.ones_like(r)
beta_ref = 15
p_ref = 2 * np.pi * 0.75 * R * np.tan(np.radians(beta_ref))
beta = lambda r : np.arctan(p_ref/(2 * np.pi * r))
Coeff = lambda a : naca16_509_m06(a, "Homework1/doc/naca16-509-m06_clcd.txt")
twist = 10
s2 = BEM_Solver(N_blades, R, R_hub, c, beta, Coeff, N_elem, tol)
s2_twist = BEM_Solver(N_blades, R, R_hub, c, beta, Coeff, N_elem, tol, twist = twist - beta_ref)


# ------ CREATE SOLVER ------
N = 50
J_array = np.linspace(1e-6, 0.8, num=N, endpoint=True)
J_array_twist = np.linspace(1e-6, 0.6, num=N, endpoint=True)
k_T = np.zeros_like(J_array)
k_Q = np.zeros_like(J_array)
k_P = np.zeros_like(J_array)
eta = np.zeros_like(J_array)
eta_2 = np.zeros_like(J_array)
for i, J in enumerate(J_array):
    print(i)
    s2.J = J
    s2.solve(0.3)
    (k_T[i], k_Q[i], k_P[i], eta[i]) = s2.get_coeff()

for i, J in enumerate(J_array_twist):
    print(i)
    s2_twist.J = J
    s2_twist.solve(0.3)
    (_,_,_, eta_2[i]) = s2_twist.get_coeff()

with open('Homework1/doc/Verification.txt') as f:
    f.readline()
    l = f.readlines()
    length = len(l)
    J_arr = np.zeros(length)
    k_T_corr = np.zeros(length)
    k_Q_corr = np.zeros(length)
    k_P_corr = np.zeros(length)
    eta_corr = np.zeros_like(J_arr)
    for i in range(length):
        J_arr[i] = l[i].split()[0]
        eta_corr[i] = l[i].split()[4]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.grid()
ax.plot(J_array, eta, 'k')
ax.plot(J_array_twist, eta_2, 'k--')
# ax.plot(J_arr, eta_corr, 'k--')
plt.show()
