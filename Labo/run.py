import numpy as np
from matplotlib import pyplot as plt

arr = np.loadtxt('Labo/data/Labo_Result_2023_12_11_10_04.txt')

min = arr[:, 1]
sec = arr[:, 2]
t = 60 * min + sec
t = t - t[0]

ps2 = arr[:, 3]
ps3 = arr[:, 4]
pt3 = arr[:, 5]
pt4 = arr[:, 6]
pt5 = arr[:, 7]
Tt3 = arr[:, 8]
Tt4 = arr[:, 9]
Tt5 = arr[:, 10]
Tt6 = arr[:, 12]
Thrust = arr[:, 14]
RPM = arr[:, 15]
m_dot = arr[:, 19]

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.plot(t, pt3, 'k-')
ax.grid()


plt.show()
