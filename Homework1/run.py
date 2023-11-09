import numpy as np
from matplotlib import pyplot as plt
from doc.stdatm import stdatm
from doc.naca16_509_m06_clcd import naca16_509_m06

# ------ SOLVER CLASS ------
class BEM_Solver:
    def __init__(self, N_blades, R, R_hub, c, beta, wing_coeff, N_elem, tol, twist:float = 0., wind_turbine : bool = False, end : float = 5, omega : float = 0.3) -> None:
        self.N_blades = N_blades
        self.R = R
        self.R_hub = R_hub
        self.h = (R - R_hub)/N_elem
        self.r = np.linspace(R_hub + self.h/2, R - self.h/2, endpoint=True, num=N_elem)

        self.c = c(self.r)
        self.beta = beta(self.r) + np.radians(twist)
        self.wing_coeff = wing_coeff
        self.N_elem = N_elem
        self.tol = tol

        self.a = np.zeros(N_elem)
        self.a_prime = np.zeros(N_elem)

        self.end = end
        self.omega = omega

        self.solved = False
        self.turbine = wind_turbine
        self.coeff = None

    def solve(self) -> None:
        omega = self.omega
        if self.coeff is None: raise(AssertionError("You should set a value for J"))
        a_k = 0
        a_prime_k = 0
        for i in range(self.N_elem):
            a_k_1 = a_k + 10 * self.tol
            a_prime_k_1 = a_prime_k + 10 * self.tol
            sigma = self.N_blades * self.c[i]/(2 * np.pi * self.r[i]) 
            while np.abs(a_k - a_k_1) >= self.tol and np.abs(a_prime_k - a_prime_k_1) >= self.tol:
                if(self.turbine):
                    phi = np.arctan2(self.coeff * self.R/(np.pi * self.r[i]) * (1 - a_k), (1 + a_prime_k))
                else:
                    phi = np.arctan2(self.coeff * self.R/(np.pi * self.r[i]) * (1 + a_k), (1 - a_prime_k))

                alpha = self.beta[i] - phi
                C_L, C_D = self.wing_coeff(alpha)
                c_N = C_L * np.cos(phi) - C_D * np.sin(phi)
                c_T = C_L * np.sin(phi) + C_D * np.cos(phi)
                a_k_1 = a_k
                a_prime_k_1 = a_prime_k

                if self.turbine:
                    a_k = (1 - omega) * a_k_1 + omega * sigma * c_N * (1 - a_k_1)/(2 * (1 - np.cos(2 * phi)))
                    a_prime_k = (1 - omega) * a_prime_k_1 + omega * sigma * c_T * (1 + a_prime_k_1)/(2 * np.sin(2 * phi))
                else:
                    a_k = (1 - omega) * a_k_1 + omega * sigma * c_N * (1 + a_k_1)/(2 * (1 - np.cos(2 * phi)))
                    a_prime_k = (1 - omega) * a_prime_k_1 + omega * sigma * c_T * (1 - a_prime_k_1)/(2 * np.sin(2 * phi))
            
            self.a[i] = a_k
            self.a_prime[i] = a_prime_k
        self.solved = True

    def get_coeff(self):
        if not self.solved:
            raise(AssertionError("The solve function should be called first"))
        k_T = self.coeff*self.coeff * np.pi/(self.R * self.R) * np.trapz(self.a * (1 + self.a) * self.r, dx=self.h)
        k_Q = self.coeff * np.pi * np.pi/(2 * self.R**4) * np.trapz(self.a_prime * (1 + self.a) * np.power(self.r, 3), dx=self.h)
        k_P = 2 * np.pi * k_Q
        eta = self.coeff * k_T/k_P
        return (k_T, k_Q, k_P, eta)
    
def compute_arrays(solver : BEM_Solver, N : int = 50, tol : float = 0.01):
    start = 0.1
    end = solver.end
    J_arr = np.linspace(start, end, num=N, endpoint=True)
    k_T, k_Q, k_P, eta = np.nan * np.ones((4, N))
    for i, J in enumerate(J_arr):
        print("Solving for J = %.4f  (i = %d)"%(J, i))
        solver.coeff = J
        solver.solve()
        (k_T[i], k_Q[i], k_P[i], eta[i]) = solver.get_coeff()
        if k_T[i] < 0:
            if(abs(k_T[i]) > tol):
                if(k_T[i] < 0): print(f"Consider reducing the range or increasing the number of samples, end closer to {J + k_T[i] * (J_arr[i-1] - J)/(k_T[i] - k_T[i-1])}")
                else: print("End reduced too much")
            break
    return J_arr, k_T, k_Q, k_P, eta



# ------ DATA FLAT PLATE ------
tol = 1e-12
N_elem = 300
R = 1/2
R_hub = 0.25/2
N_blades = 2
c = lambda r : 0.15 * np.ones_like(r)
beta = lambda r : np.radians(25) * np.ones_like(r)
Coeff = lambda a : (2 * np.pi * a, 0)
s1 = BEM_Solver(N_blades, R, R_hub, c, beta, Coeff, N_elem, tol, end = 1.0309, omega=0.5)


# ------ DATA HAMILTON-STANDARD 24D50 - 6813 ------
tol = 1e-8
N_elem = 50
R = 3.4/2
R_hub = 0.45/2
N_blades = 4
c = lambda r : 0.15 * np.ones_like(r)
beta_ref = 15
p_ref = 2 * np.pi * 0.75 * R * np.tan(np.radians(beta_ref))
beta = lambda r : np.arctan(p_ref/(2 * np.pi * r))
Coeff = lambda a : naca16_509_m06(a, "Homework1/doc/naca16-509-m06_clcd.txt")
s2 = BEM_Solver(N_blades, R, R_hub, c, beta, Coeff, N_elem, tol, end=0.7942)

twists = np.array([(i+1) * 10 for i in range(6)]) - beta_ref
ends = np.array([0.584, 1.021357, 1.5377, 2.217, 1.0102, 0.615])
omegas = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])

s2_with_pitch = np.empty_like(twists, dtype=BEM_Solver)
for i in range(len(twists)):
    s2_with_pitch[i] = BEM_Solver(N_blades, R, R_hub, c, beta, Coeff, N_elem, tol, twist = twists[i], end=ends[i], omega=omegas[i])


# ------ FIRST PLOT ------
results = compute_arrays(s1, N = 50, tol=1e-5)
labels = [r'$k_T$', r'$k_Q$', r'$k_P$', r'$\eta$']

with open('Homework1/doc/Verification.txt') as f:
    f.readline()
    l = f.readlines()
    length = len(l)
    correct_results = np.zeros((5, length))
    for i in range(length):
        correct_results[:, i] = l[i].split()

fig, ax = plt.subplots(2, 2, figsize = (15, 8))

fig.suptitle('Verification of the solver using the given data')

for k in range(4):
    i = k//2
    j = k%2
    ax[i, j].grid()
    ax[i, j].set_xlabel('J')
    ax[i, j].set_ylabel(labels[k], size=12)
    ax[i, j].plot(results[0], results[k+1], 'k-', label='Computed values')
    ax[i, j].plot(correct_results[0], correct_results[k+1], 'k--', label='Given values')
    ax[i, j].legend()

# ------ SECOND PLOT ------
results = compute_arrays(s2_with_pitch[5], N = 50, tol=1e-5)
labels = [r'$k_T$', r'$k_Q$', r'$k_P$', r'$\eta$']

fig, ax = plt.subplots(2, 2, figsize = (15, 8))

fig.suptitle('Coefficients curves for the P-51D proppeler')

for k in range(4):
    i = k//2
    j = k%2
    ax[i, j].grid()
    ax[i, j].set_xlabel('J')
    ax[i, j].set_ylabel(labels[k], size=12)
    ax[i, j].plot(results[0], results[k+1], 'k-')

plt.show()
