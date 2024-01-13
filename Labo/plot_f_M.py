import numpy as np
from matplotlib import pyplot as plt


M = np.linspace(0, 3, num = 1000)


f = lambda M, gamma : M * np.power(2/(gamma + 1) * (1 + (gamma - 1)/2 * M*M), -(gamma + 1)/(2 * (gamma - 1)))

g = np.linspace(1.3, 1.5, num=4)

c = ['k-', 'r-', 'b-', 'g-', 'k--', 'r--', 'b--', 'g--']

fig, ax = plt.subplots()
for i, gam in enumerate(g):
    ax.plot(M, f(M, gam), c[i], label=r'$\gamma = %.2f$'%(gam))

ax.grid()
ax.set_xlabel('M')
ax.set_ylabel('f(M)')
ax.legend()

fig.savefig('images/f_M.pdf', format='pdf')
plt.show()




