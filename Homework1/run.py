import numpy as np
from matplotlib import pyplot as plt
from doc.stdatm import stdatm
from doc.naca16_509_m06_clcd import naca16_509_m06
from scipy.interpolate import CubicSpline

PLOTS = [False, True, True]

# ------ SOLVER CLASS ------
class BEM_Solver:
    def __init__(self, N_blades, R, R_hub, c, beta, wing_coeff, N_elem, tol, twist:float = 0., wind_turbine : bool = False, end : float = 10, start : float = 0.1, omega : float = 0.3, init_val_a : float = 0, init_val_a_prime : float = 0) -> None:
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
        self.start = start
        self.omega = omega
        self.init_val_a = init_val_a
        self.init_val_a_prime = init_val_a_prime

        self.solved = False
        self.turbine = wind_turbine
        self.coeff = None

    def solve(self) -> None:
        omega = self.omega
        if self.coeff is None: raise(AssertionError("You should set a value for J of lambda"))
        a_k = self.init_val_a
        a_prime_k = self.init_val_a_prime
        for i in range(self.N_elem):
            a_k_1 = a_k + 10 * self.tol
            a_prime_k_1 = a_prime_k + 10 * self.tol
            sigma = self.N_blades * self.c[i]/(2 * np.pi * self.r[i]) 
            while np.abs(a_k - a_k_1) >= self.tol and np.abs(a_prime_k - a_prime_k_1) >= self.tol:
                if self.turbine:
                    phi = np.arctan2(self.R/(self.coeff * self.r[i]) * (1 - a_k), (1 + a_prime_k))
                    alpha = phi - self.beta[i]
                else:
                    phi = np.arctan2(self.coeff * self.R/(np.pi * self.r[i]) * (1 + a_k), (1 - a_prime_k))
                    alpha = self.beta[i] - phi

                C_L, C_D = self.wing_coeff(alpha)

                if self.turbine:
                    c_N = C_L * np.cos(phi) + C_D * np.sin(phi)
                    c_T = C_L * np.sin(phi) - C_D * np.cos(phi)
                else:
                    c_N = C_L * np.cos(phi) - C_D * np.sin(phi)
                    c_T = C_L * np.sin(phi) + C_D * np.cos(phi)
                
                a_k_1 = a_k
                a_prime_k_1 = a_prime_k

                if self.turbine:
                    a_new = sigma/2 * (1 - a_k)/(1 - np.cos(2 * phi)) * c_N
                    a_prime_new = sigma/2 * (1 + a_prime_k)/np.sin(2 * phi) * c_T
                else:
                    a_new = sigma/2 * (1 + a_k)/(1 - np.cos(2 * phi)) * c_N
                    a_prime_new = sigma/2 * (1 - a_prime_k)/np.sin(2 * phi) * c_T

                a_k = (1 - omega) * a_k_1 + omega * a_new
                a_prime_k = (1 - omega) * a_prime_k_1 + omega * a_prime_new

            self.a[i] = a_k
            self.a_prime[i] = a_prime_k
        self.solved = True

    def get_coeff(self):
        if not self.solved:
            raise(AssertionError("The solve function should be called first"))
        
        if not self.turbine:
            k_T = self.coeff*self.coeff * np.pi/(self.R * self.R) * np.trapz(self.a * (1 + self.a) * self.r, dx=self.h)
            k_Q = self.coeff * np.pi * np.pi/(2 * self.R**4) * np.trapz(self.a_prime * (1 + self.a) * np.power(self.r, 3), dx=self.h)
            k_P = 2 * np.pi * k_Q
            eta = self.coeff * k_T/k_P
            return (k_T, k_Q, k_P, eta)
        else:
            C_T = 8/(self.R * self.R) * np.trapz(self.a * (1 - self.a) * self.r, dx=self.h)
            C_P = 8 * self.coeff*self.coeff/(self.R**4) * np.trapz(self.a_prime * (1 - self.a) * self.r, dx=self.h)
            return C_T,C_P
        
def find_max(x, y):
    c = CubicSpline(x, y)
    x_test = np.linspace(x[0], x[-1], num=500, endpoint=True)
    y_test = c(x_test)

    i = np.argmax(y_test)

    return x_test[i], y_test[i]
    
def compute_arrays_prop(solver : BEM_Solver, N : int = 50, tol : float = 0.01):
    if solver.turbine: raise(ValueError("The solver should be from a propeller"))
    start = solver.start
    end = solver.end
    first_a = 0
    first_a_prime = 0
    J_arr = np.linspace(start, end, num=N, endpoint=True)
    k_T, k_Q, k_P, eta = np.nan * np.ones((4, N))
    for i, J in enumerate(J_arr):
        print("Solving for J = %.4f  (i = %d)"%(J, i))
        solver.coeff = J
        solver.init_val_a = first_a
        solver.init_val_a_prime = first_a_prime
        solver.solve()
        first_a = solver.a[0]
        first_a_prime = solver.a_prime[0]
        (k_T[i], k_Q[i], k_P[i], eta[i]) = solver.get_coeff()
        if k_T[i] < 0:
            if(abs(k_T[i]) > tol):
                if(k_T[i] < 0): print(f"Consider reducing the range or increasing the number of samples, end closer to {J + k_T[i] * (J_arr[i-1] - J)/(k_T[i] - k_T[i-1])}")
                else: print("End reduced too much")
            break
    J_opt, eta_opt = find_max(J_arr, eta)
    return J_arr, k_T, k_Q, k_P, eta, J_opt, eta_opt

def compute_arrays_turb(solver : BEM_Solver, N : int = 50, tol : float = 0.01):
    if not solver.turbine: raise(ValueError("The solver should be from a wind turbine"))
    start = solver.start
    end = solver.end
    first_a = 0
    first_a_prime = 0
    lam_arr = np.linspace(end, start, num=N, endpoint=True)
    C_T, C_P = np.nan * np.ones((2, N))
    for i, lam in enumerate(lam_arr):
        print("Solving for lambda = %.4f  (i = %d)"%(lam, i))
        solver.coeff = lam
        solver.init_val_a = first_a
        solver.init_val_a_prime = first_a_prime
        solver.solve()
        first_a = solver.a[0]
        first_a_prime = solver.a_prime[0]
        (C_T[i], C_P[i]) = solver.get_coeff()
        if C_P[i] < 0:
            if(abs(C_P[i]) > tol):
                if(C_P[i] < 0): print(f"Consider reducing the range or increasing the number of samples, end closer to {lam + C_P[i] * (lam_arr[i-1] - lam)/(C_P[i] - C_P[i-1])}")
                else: print("End reduced too much")
            break
        if np.max(solver.a) > 0.5:
            print(f"Problem for lambda = {lam}")

    lam_opt, C_P_opt = find_max(np.flip(lam_arr), np.flip(C_P))

    return lam_arr, C_T, C_P, lam_opt, C_P_opt


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

twists_prop = np.array([(i+1) * 10 for i in range(6)]) - beta_ref
ends = np.array([0.584, 1.021357, 1.5377, 2.217, 3.2088, 4.8658])
starts = np.array([0.1, 0.1, 0.1, 0.1, 1.2, 2.2])

s2_with_pitch = np.empty_like(twists_prop, dtype=BEM_Solver)
for i in range(len(twists_prop)):
    s2_with_pitch[i] = BEM_Solver(N_blades, R, R_hub, c, beta, Coeff, N_elem, tol, twist = twists_prop[i], end=ends[i], start=starts[i], omega=0.7)

# ------ TURBINE ------
twists_turb = np.array([5 + 3 * i for i in range(6)]) - beta_ref
ends = np.array([4.47, 5.56, 7.3, 6.2, 5.3, 4.5])
starts = np.array([3.583483312073389, 3.12657, 2.73133, 2.388287, 2.087355, 1.807582296586886])

s3_with_pitch = np.empty_like(twists_turb, dtype=BEM_Solver)
for i in range(len(twists_turb)):
    s3_with_pitch[i] = BEM_Solver(N_blades, R, R_hub, c, beta, Coeff, N_elem, tol, twist = twists_turb[i], end=ends[i], start=starts[i], omega=0.7, wind_turbine=True)

fine_twists_turb = np.array([i for i in range(10)]) - beta_ref
ends = np.array([3, 5.56, 7.3, 4, 4.248, 4.47, 4.67, 4.81, 5.56, 6])
starts = np.array([3.583483312073389, 3.12657, 2.73133, 2.388287, 3.75, 3.583483312073389, 3.5, 3.5, 3.12657, 3])

s4_fine_pitch = np.empty_like(fine_twists_turb, dtype=BEM_Solver)
for i in range(len(fine_twists_turb)):
    s4_fine_pitch[i] = BEM_Solver(N_blades, R, R_hub, c, beta, Coeff, N_elem, tol, twist = fine_twists_turb[i], end=ends[i], start=starts[i], omega=0.7, wind_turbine=True)

# ------ FIRST PLOT ------
if PLOTS[0]:
    results = compute_arrays_prop(s1, N = 50, tol=1e-5)
    labels = [r'$k_T$', r'$k_Q$', r'$k_P$', r'$\eta_P$']

    with open('Homework1/doc/Verification.txt') as f:
        f.readline()
        l = f.readlines()
        length = len(l)
        correct_results = np.zeros((5, length))
        for i in range(length):
            correct_results[:, i] = l[i].split()

    fig, ax = plt.subplots(2, 2, figsize = (15, 8))

    fig.suptitle('Verification of the solver using the given data', y = 0.91)

    for k in range(4):
        i = k//2
        j = k%2
        ax[i, j].grid()
        ax[i, j].set_xlabel('J')
        ax[i, j].set_ylabel(labels[k], size=12)
        ax[i, j].plot(results[0], results[k+1], 'k-', label='Computed values')
        ax[i, j].plot(correct_results[0], correct_results[k+1], 'k--', label='Given values')
        ax[i, j].legend()

    fig.savefig('Homework1/images/flat_plate.eps', format='eps')

# ------ SECOND PLOT ------
if PLOTS[1]:
    fig, ax = plt.subplots(2, 2, figsize = (15, 10))
    colors = ['k', 'r', 'g', 'b', 'm', 'c']
    fig.suptitle('Coefficients curves for the P-51D propeller', y = 0.91)

    for s in range(len(s2_with_pitch)):
        results = compute_arrays_prop(s2_with_pitch[s], N = 50, tol=1e-5)
        labels = [r'$k_T$', r'$k_Q$', r'$k_P$', r'$\eta_P$']

        for k in range(4):
            i = k//2
            j = k%2
            ax[i, j].plot(results[0], results[k+1], colors[s], label= r'$\beta_{0.75R} = %d ^{\circ}$'%(twists_prop[s] + beta_ref))
        ax[1, 1].scatter(results[5], results[6], c=colors[s])

    for k in range(4):
        i = k//2
        j = k%2
        ax[i, j].grid()
        ax[i, j].set_xlabel('J')
        ax[i, j].set_ylabel(labels[k], size=12)
        ax[i, j].legend()

    fig.savefig('Homework1/images/propeller.eps', format='eps')

# ------ THIRD PLOT ------
if PLOTS[2]:
    fig, ax = plt.subplots(1, 2, figsize = (15, 8))

    fig.suptitle('Coefficients curves for the P-51D propeller as a wind turbine', y = 0.91)

    for s in range(len(s3_with_pitch)):
        results = compute_arrays_turb(s3_with_pitch[s], tol=1e-5)
        labels = [r'$C_T$', r'$C_P$']

        for i in range(2):
            ax[i].plot(results[0], results[i+1], colors[s], label= r'$\beta_{0.75R} = %d ^{\circ}$'%(twists_turb[s] + beta_ref))
        ax[1].scatter(results[3], results[4], c=colors[s])

    for i in range(2):
        ax[i].grid()
        ax[i].set_xlabel(r'$\lambda$')
        ax[i].set_ylabel(labels[i], size=12)
        ax[i].legend()

    ax[1].axhline(16/27, color='k', linestyle='--')
    fig.savefig('Homework1/images/turbine.eps', format='eps')


# 6 : 0.5300185763104968
# 7 : 0.5258044587611028
# 8 : 0.5307414263823855
# 9 : 0.5152135149592543

plt.show()
