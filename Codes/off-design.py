import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

gamma = 1.4

Ae_A8 = np.linspace(1, 15, num=200)
F_u = np.zeros_like(Ae_A8)
F_p = np.zeros_like(Ae_A8)

for i, A in enumerate(Ae_A8): 
    func = lambda M : 1/A - np.power(2/(gamma + 1) * (1 + (gamma - 1)/2 * M*M), -(gamma + 1)/(2 * (gamma - 1)))*M
    Me = fsolve(func, 2)[0]
    Te_T0 = 5.67/(1 + (gamma - 1)/2 * Me*Me)
    pe_p0 = 86.22/np.power(1 + (gamma - 1)/2 * Me*Me, gamma/(gamma - 1))
    ue_u0 = Me/3 * np.sqrt(Te_T0)
    F_u[i] = gamma * 9 * (ue_u0 - 1)
    F_p[i] = A/7 * (pe_p0 - 1)

F = F_u + F_p


fig, ax = plt.subplots()
ax.plot(Ae_A8, F, 'k-')
ax.plot(Ae_A8, F_u, 'g--')
ax.plot(Ae_A8, F_p, 'r--')
ax.grid()

plt.show()
