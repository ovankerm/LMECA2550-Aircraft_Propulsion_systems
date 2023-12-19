import numpy as np
from matplotlib import pyplot as plt

# ------ ENGINE DATA ------
pi_c = 11.3
Tt4 = 1620
Tt7 = 2170
LHV = 43.19e6
ec = 0.9
et = 0.9

# ------ FLIGHT CONDITIONS ------
p0 = 10465
T0 = 187.1
rho0 = 0.195
R = p0/(T0 * rho0)
gamma = 1.4

# ------ CALCULATED COEFFICIENTS ------
tau_lam = Tt4/T0
tau_lam_AB = Tt7/T0
tau_c = np.power(pi_c, (gamma - 1)/(gamma * ec))
c0 = np.sqrt(gamma * R * T0)
cp = gamma/(gamma - 1) * R


# ------ FUNCTIONS ------
pi_r = lambda M0 : np.power(1 + (gamma - 1)/2 * np.power(M0, 2), gamma/(gamma - 1))
tau_r = lambda M0 : 1 + (gamma - 1)/2 * np.power(M0, 2)
tau_t = lambda M0 : 1 - tau_r(M0)/tau_lam * (tau_c - 1)
tau_b = lambda M0 : tau_lam/(tau_r(M0) * tau_c)
tau_AB = lambda M0 : tau_lam_AB/(tau_lam * tau_t(M0))
specific_thrust = lambda M0 : c0 * (np.sqrt(2/(gamma - 1) * (tau_r(M0) * np.power(tau_c, ec) * np.power(tau_t(M0), 1/et) - 1) * tau_lam_AB/(tau_r(M0) * np.power(tau_c, ec) * np.power(tau_t(M0), 1/et))) - M0)
air_fuel_ratio = lambda M0 : cp * T0/LHV * (tau_r(M0) * tau_c * (tau_b(M0) - 1) + tau_lam * tau_t(M0) * (tau_AB(M0) - 1))
TSFC = lambda M0 : air_fuel_ratio(M0)/specific_thrust(M0)
eta_T = lambda M0 : (tau_lam_AB * (1 - 1/(np.power(tau_c, ec) * np.power(tau_t(M0), 1/et) * tau_r(M0))) - tau_r(M0) + 1)/(tau_lam * (1 - tau_t(M0)) + tau_lam_AB - tau_r(M0) * tau_c)
eta_P = lambda M0 : 2 * M0/(np.sqrt(2/(gamma - 1) * (tau_r(M0) * np.power(tau_c, ec) * np.power(tau_t(M0), 1/et) - 1) * tau_lam_AB/(tau_r(M0) * np.power(tau_c, ec) * np.power(tau_t(M0), 1/et))) + M0)




M = np.linspace(0, 4.5, num=100)

ar = np.argmax(eta_T(M))
print(M[ar], eta_T(M)[ar])

# fig, ax = plt.subplots(1, 1, figsize=(16, 9))
# ax.plot(M, specific_thrust(M), 'k-')
# ax.grid()
# ax.set_title('Specific thrust', size = 20)
# ax.set_xlabel(r'$M_0$', size = 15)
# ax.set_ylabel(r'$\frac{F}{\dot{m_0}}$', size = 25)
# fig.savefig("Homework2/images/thrust.eps", format='eps')

# fig, ax = plt.subplots(1, 1, figsize=(16, 9))
# ax.plot(M, TSFC(M), 'k-')
# ax.grid()
# ax.set_title('Thrust specific fuel consumption', size = 20)
# ax.set_xlabel(r'$M_0$', size = 15)
# ax.set_ylabel(r'$TSFC$', size = 15)
# fig.savefig("Homework2/images/tsfc.eps", format='eps')

# fig, ax = plt.subplots(1, 1, figsize=(16, 9))
# ax.plot(M, air_fuel_ratio(M), 'k-')
# ax.grid()
# ax.set_title('Total fuel/air ratio', size = 20)
# ax.set_xlabel(r'$M_0$', size = 15)
# ax.set_ylabel(r'$f$', size = 15)
# fig.savefig("Homework2/images/f.eps", format='eps')

# fig, ax = plt.subplots(1, 1, figsize=(16, 9))
# ax.plot(M, eta_T(M), 'k-')
# ax.grid()
# ax.set_title('Thermal efficiency', size = 20)
# ax.set_xlabel(r'$M_0$', size = 15)
# ax.set_ylabel(r'$\eta_T$', size = 15)
# fig.savefig("Homework2/images/eta_T.eps", format='eps')

# fig, ax = plt.subplots(1, 1, figsize=(16, 9))
# ax.plot(M, eta_P(M), 'k-')
# ax.grid()
# ax.set_title('Propulsive efficiency', size = 20)
# ax.set_xlabel(r'$M_0$', size = 15)
# ax.set_ylabel(r'$\eta_P$', size = 15)
# fig.savefig("Homework2/images/eta_P.eps", format='eps')

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(M, tau_b(M), 'k-')
ax.grid()
ax.set_title('Burner total temperature ratio', size = 20)
ax.set_xlabel(r'$M_0$', size = 15)
ax.set_ylabel(r'$\tau_b$', size = 15)
fig.savefig("Homework2/images/tau_b.eps", format='eps')


M0R = 2
tau_tR = tau_t(M0R)

def offdesign(M0):
    tau_r_ = tau_r(M0)
    pi_c_tilde = np.power((1 + tau_lam/tau_r_ * (1 - tau_tR)), gamma * ec/(gamma - 1))

    pi_c_ = np.where(pi_c_tilde <= pi_c, pi_c_tilde, pi_c)
    tau_c_ = np.power(pi_c_, (gamma - 1)/(gamma * ec))
    tau_lam_ = np.where(pi_c_tilde <= pi_c, tau_lam, tau_r_ * (tau_c_ - 1)/(1 - tau_tR))

    tau_b_ = tau_lam_/(tau_r_ * tau_c_)
    tau_AB_ = tau_lam_AB/(tau_lam_ * tau_tR)

    thrust = c0 * (np.sqrt(2/(gamma - 1) * (tau_r_ * np.power(tau_c_, ec) * np.power(tau_tR, 1/et) - 1) * tau_lam_AB/(tau_r_ * np.power(tau_c_, ec) * np.power(tau_tR, 1/et))) - M0)
    f = cp * T0/LHV * (tau_r_ * tau_c_ * (tau_b_ - 1) + tau_lam_ * tau_tR * (tau_AB_ - 1))

    tsfc = f/thrust

    eta_T_ = (tau_lam_AB * (1 - 1/(np.power(tau_c_, ec) * np.power(tau_tR, 1/et) * tau_r_)) - tau_r_ + 1)/(tau_lam_ * (1 - tau_tR) + tau_lam_AB - tau_r_ * tau_c_)
    eta_P_ = 2 * M0/(np.sqrt(2/(gamma - 1) * (tau_r_ * np.power(tau_c_, ec) * np.power(tau_tR, 1/et) - 1) * tau_lam_AB/(tau_r_ * np.power(tau_c_, ec) * np.power(tau_tR, 1/et))) + M0)
    return (thrust, f, tsfc, eta_T_, eta_P_, tau_b_)


(thrust, f, tsfc, eta_T_, eta_P_, tau_b_) = offdesign(M)


fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(M, thrust, 'k--', label='off-design')
ax.plot(M, specific_thrust(M), 'k-', label='on-design')
ax.scatter(2, specific_thrust(2), c = 'r', label='design point')
ax.grid()
ax.legend()
ax.set_title('Specfic thrust', size = 20)
ax.set_xlabel(r'$M_0$', size = 15)
ax.set_ylabel(r'$\frac{F}{\dot{m_0}}$', size = 25)
fig.savefig("Homework2/images/off_thrust.eps", format='eps')

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(M, tsfc, 'k--', label='off-design')
ax.plot(M, TSFC(M), 'k-', label='on-design')
ax.scatter(2, TSFC(2), c = 'r', label='design point')
ax.grid()
ax.legend()
ax.set_title('Thrust specific fuel consumption', size = 20)
ax.set_xlabel(r'$M_0$', size = 15)
ax.set_ylabel(r'$TSFC$', size = 15)
fig.savefig("Homework2/images/off_tsfc.eps", format='eps')

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(M, f, 'k--', label='off-design')
ax.plot(M, air_fuel_ratio(M), 'k-', label='on-design')
ax.scatter(2, air_fuel_ratio(2), c = 'r', label='design point')
ax.grid()
ax.legend()
ax.set_title('Total fuel/air ratio', size = 20)
ax.set_xlabel(r'$M_0$', size = 15)
ax.set_ylabel(r'$f$', size = 15)
fig.savefig("Homework2/images/off_f.eps", format='eps')

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(M, eta_T_, 'k--', label='off-design')
ax.plot(M, eta_T(M), 'k-', label='on-design')
ax.scatter(2, eta_T(2), c = 'r', label='design point')
ax.grid()
ax.legend()
ax.set_title('Thermal efficiency', size = 20)
ax.set_xlabel(r'$M_0$', size = 15)
ax.set_ylabel(r'$\eta_T$', size = 15)
fig.savefig("Homework2/images/off_eta_T.eps", format='eps')

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(M, eta_P_, 'k--', label='off-design')
ax.plot(M, eta_P(M), 'k-', label='on-design')
ax.scatter(2, eta_P(2), c = 'r', label='design point')
ax.grid()
ax.legend()
ax.set_title('Propulsive efficiency', size = 20)
ax.set_xlabel(r'$M_0$', size = 15)
ax.set_ylabel(r'$\eta_P$', size = 15)
fig.savefig("Homework2/images/off_eta_P.eps", format='eps')

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(M, tau_b_, 'k-', label='off-design')
ax.grid()
ax.set_title('Burner total temperature ratio', size = 20)
ax.set_xlabel(r'$M_0$', size = 15)
ax.set_ylabel(r'$\tau_b$', size = 15)
fig.savefig("Homework2/images/off_tau_b.eps", format='eps')


plt.show()