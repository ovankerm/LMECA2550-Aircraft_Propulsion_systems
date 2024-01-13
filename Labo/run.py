import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve


# ------ DATA ------
regime_indices = {0 : (0, 14), 36000: (15, 34), 50000 : (35, 44), 60000 : (45, 69), 70000 : (70, 94), 80000 : (95, 119), 90000 : (120, 144), 100000 : (145, 169)}
R = 287.1
gamma = 1.4
LHV = 43.7e6 # [J/kg]
A1 = 5.168e-3
A2 = 2.818e-3
A3 = 4.112e-3
A4 = 6.323e-3
A5 = 3.519e-3
A6 = 3.318e-3

# ------ FUNCTIONS ------
def table_to_latex(table): 
    with open("data/latex.txt", "w") as f: 
        for i in range(len(table)): 
            for j in range(len(table[0])): 
                f.write(" $%.3f$ "%(table[i, j]))
                if j < len(table[0]) - 1: 
                    f.write("&")
            f.write("\\\\  \\hline \n")

def get_mean(array : np.ndarray, RPM : int):
    return np.mean(array[regime_indices[RPM][0]:regime_indices[RPM][1]+1])

def get_cp(T : float, gas : bool = False, f : float = 0):
    if 200 < T and T <= 800:
        cpa = 1.0189e3 + T*(-0.13784 + T*(1.9843e-4 + T*(4.2399e-7 - 3.7632e-10 * T)))
        if not gas: return cpa
        Bt = -3.59494e2 + T * (4.5164 + T * (2.8116e-3 + T * (-2.1709e-5 + T * (2.8689e-8 - T * 1.2263e-11))))
        return cpa + Bt * f/(f+1)
    elif 800 < T and T < 2200:
        cpa = 7.9865e2 + T*(0.5339 + T*(-2.2882e-4 + T*3.7421e-8))
        if not gas: return cpa
        Bt = 1.0888e3 + T * (-0.1416 + T * (1.916e-3 + T * (-1.2401e-6 + T * (3.0669e-10 - T * 2.6117e-14))))
        return cpa + Bt * f/(f+1)
    return 0

def get_gamma(ps : float, pt : float, Tt : float, tol: float = 1e-12, gas: bool = False, f: float = 0, init_guess : float = 1.4):
    gamma_curr = init_guess
    gamma_prev = gamma_curr + 10 * tol
    while abs(gamma_curr - gamma_prev) >= tol:
        T = Tt * np.power(ps/pt, (gamma_curr - 1)/gamma_curr)
        cp = get_cp(T, gas, f)
        gamma_prev = gamma_curr
        gamma_curr = cp/(cp - R)
    return gamma_curr

def gamma_from_A_star(A_star : float, Tt : float, Area : float, init_guess : float = 1.4, tol : float = 1e-12, gas : bool = False, f: float = 0):
    gamma_curr = init_guess
    gamma_prev = gamma_curr + 10 * tol
    M_init = 0.5
    Ts e 0
    while abs(gamma_curr - gamma_prev) >= tol:
        func = lambda M : A_star/Area - M * np.power((gamma_curr+1)/(2 * (1 + (gamma_curr-1)/2 * M*M)), (gamma_curr+1)/(2 * (gamma_curr-1)))
        M_init = fsolve(func, M_init)[0]
        Ts = Tt/(1 + (gamma_curr - 1)/2 * M_init*M_init)
        cp = get_cp(Ts, gas, f)
        gamma_prev = gamma_curr
        gamma_curr = cp/(cp - R)
    return gamma_curr, M_init, Ts

def T_from_m_dot(pt : float, Tt : float, A : float, m_dot : float, T_init : float, tol : float = 1e-12): 
    T = T_init
    T_prev = T_init + 10 * tol
    omega = 1
    rho_t = pt * 1e5/(R * Tt)
    while(abs(T_prev - T) >= tol):
        print()
        print("---------------new iter--------------")
        cp = get_cp(T)
        gamma = 1.4
        print("g =", gamma)
        print("Tt =", Tt)
        print("t = ", Tt/T)
        M = np.sqrt(2/(gamma - 1) * (Tt/T - 1))
        print("M :", M)
        rho = rho_t/np.power(1 + (gamma - 1)/2 * M*M, 1/(gamma - 1))
        T_prev = T
        T = np.power(m_dot/(rho * M * A), 2)/(gamma * R)
        T = (1 - omega) * T_prev + omega * T
        print(T)
        print()
    return T



# ------ REGIMES ------
arr = np.loadtxt('data/Labo_Result_2023_12_11_10_04SaveButton.txt')

hour = arr[:, 0]
minute = arr[:, 1]
sec = arr[:, 2]

t = 3600 * hour + 60 * minute + sec
t = t - t[0]

ps2 = arr[:, 4]
pt3 = arr[:, 5]
ps3 = arr[:, 6]
pt4 = arr[:, 7]
pt5 = arr[:, 8]
Tt3 = arr[:, 9] + 273.15
Tt4 = arr[:, 10] + 273.15
Tt5 = arr[:, 11] + 273.15
Tt6 = arr[:, 13] + 273.15
Thrust = arr[:, 15]
RPM = arr[:, 16]
m_dot = arr[:, 20] * 1e-3

#--- Params ---
m_dot_offset = get_mean(m_dot, 0)
T_offset = get_mean(Thrust, 0)
T0 = np.mean(np.array([get_mean(Tt3, 0), get_mean(Tt4, 0), get_mean(Tt5, 0), get_mean(Tt6, 0)]))
p0 = np.mean(np.array([get_mean(pt4, 0), get_mean(ps3, 0), get_mean(pt3, 0), get_mean(ps2, 0), get_mean(pt5, 0)]))
m_dot -= m_dot_offset

cp0 = get_cp(T0, False)
state0 = np.array([p0, p0, T0, T0, 0, 0, cp0/(cp0 - R)])


def compute_states(RPM: int):
    table = np.zeros((7, 7))
    # state 0
    table[:, 0] = state0

    # state 2
    state2 = np.copy(state0)
    state2[0] = get_mean(ps2, RPM)
    state2[6] = get_gamma(state2[0], state2[1], state2[3], init_guess = state0[6])
    state2[2] = state2[3] * np.power(state2[0]/state2[1], (state2[6] - 1)/state2[6])
    M = np.sqrt(2/(state2[6] - 1) * (state2[3]/state2[2] - 1))
    state2[4] = M * np.sqrt(state2[6] * R * state2[2])
    table[:, 2] = state2
    rho2 = state2[0] * 1e5/(R * state2[2])
    cp2 = state2[6] * R/(state2[6] - 1)
    m_dot_air = state2[4] * rho2 * A2

    # state 1
    state1 = np.copy(state2)
    A_star = A2 * M * np.power((state2[6] + 1)/(2 * (1 + (state2[6] - 1)/2 * M*M)), (state2[6] + 1)/(2 * (state2[6] - 1)))
    state1[6], M1, state1[2] = gamma_from_A_star(A_star, state1[3], A1, init_guess = state0[6])
    state1[0] = state1[1] * np.power(1 + (state1[6] - 1)/2 * M1*M1, state1[6]/(state1[6] - 1))
    state1[4] = M1 * np.sqrt(state1[6] * R * state1[2])
    table[:, 1] = state1

    # state 3
    state3 = np.zeros(7)
    state3[0] = get_mean(ps3, RPM)
    state3[1] = get_mean(pt3, RPM)
    state3[3] = get_mean(Tt3, RPM)
    state3[6] = get_gamma(state3[0], state3[1], state3[3], init_guess = state2[6])
    state3[2] = state3[3] * np.power(state3[0]/state3[1], (state3[6] - 1)/state3[6])
    rho3 = state3[0] * 1e5/(R * state3[2])
    state3[4] = m_dot_air/(rho3 * A3)
    cp3 = state3[6] * R/(state3[6] - 1)
    state3[5] = state2[5] + (cp3 - cp2) + (cp2 - (cp3 - cp2)/(state3[2] - state2[2]) * state2[2]) * np.log(state3[2]/state2[2]) - R * np.log(state3[0]/state2[0])
    table[:, 3] = state3

    # state 4
    state4 = np.zeros(7)
    state4[1] = get_mean(pt4, RPM)
    state4[0] = get_mean(ps3, RPM)
    state4[3] = get_mean(Tt4, RPM)
    state4[6] = get_gamma(state4[0], state4[1], state4[3], init_guess = state3[6])
    f = get_mean(m_dot, RPM)/m_dot_air
    m_dot_fuel = get_mean(m_dot, RPM)
    cp4 = (m_dot_air * cp3 * state3[3] + m_dot_fuel * LHV)/(state4[3] * (m_dot_air + m_dot_fuel))
    state4[6] = cp4/(cp4 - R)
    func = lambda T : cp4 - get_cp(T, True, f)
    state4[2] = fsolve(func, 300)[0]
#    state4[2] = state4[3] * np.power(state4[0]/state4[1], (state4[6] - 1)/state4[6])
    state4[0] = state4[1] * np.power(state4[2]/state4[3], state4[6]/(state4[6] - 1))
    M4 = np.sqrt(2/(state4[6] - 1) * (np.power(state4[1]/state4[0], (state4[6] - 1)/state4[6]) - 1))
    state4[4] = M4 * np.sqrt(state4[6] * R * state4[2])
#    cp4 = state4[6] * R/(state4[6] - 1)
    state4[5] = state3[5] + (cp4 - cp3) + (cp3 - (cp4 - cp3)/(state4[2] - state3[2]) * state3[2]) * np.log(state4[2]/state3[2]) - R * np.log(state4[0]/state3[0])
    table[:, 4] = state4

    # state 6
    state6 = np.copy(state0)
    state6[1] = get_mean(pt5, RPM)
    state6[3] = get_mean(Tt6, RPM)
    state6[6] = get_gamma(state6[0], state6[1], state6[3], init_guess = state0[6], gas=True, f=f)
    state6[2] = state6[3] * np.power(state6[0]/state6[1], (state6[6] - 1)/state6[6])
    M6 = np.sqrt(2/(state6[6] - 1) * (state6[3]/state6[2] - 1))
    state6[4] = M6 * np.sqrt(state6[6] * R * state6[2])
    cp6 = state6[6] * R/(state6[6] - 1)
    state6[5] = state4[5] + (cp6 - cp4) + (cp4 - (cp6 - cp4)/(state6[2] - state4[2]) * state4[2]) * np.log(state6[2]/state4[2]) - R * np.log(state6[0]/state4[0])
    table[:, 6] = state6

    # state 5
    state5 = np.copy(state6)
    state5[3] = get_mean(Tt5, RPM)
    A_star = A6 * M6 * np.power((state6[6] + 1)/(2 * (1 + (state6[6] - 1)/2 * M6*M6)), (state6[6] + 1)/(2 * (state6[6] - 1)))
    state5[6], M5, state5[2] = gamma_from_A_star(A_star, state5[3], A5, init_guess = state6[6], gas=True, f=f)
    state5[0] = state5[1] * np.power(1 + (state5[6] - 1)/2 * M5*M5, state5[6]/(state5[6] - 1))
    state5[4] = M5 * np.sqrt(state5[6] * R * state5[2])
    table[:, 5] = state5

    return table


table = compute_states(80000)
print(table)
#table_to_latex(table)
fig, ax = plt.subplots()
ax.scatter(table[5], table[2])
print(table[2])
print(table[3])
# ax.set_ylim(250, 2200)
plt.show()


# ------ EVOLUTION WITH TIME ------
arr = np.loadtxt('data/Labo_Result_2023_12_11_10_04.txt')

hour = arr[:, 0]
minute = arr[:, 1]
sec = arr[:, 2]

t = 3600 * hour + 60 * minute + sec
t = t - t[0]

ps2 = arr[:, 3]
pt3 = arr[:, 4]
ps3 = arr[:, 5]
pt4 = arr[:, 6]
pt5 = arr[:, 7]
Tt3 = arr[:, 8]
Tt4 = arr[:, 9]
Tt5 = arr[:, 10]
Tt6 = arr[:, 12]
Thrust = arr[:, 14]
RPM = arr[:, 15]
m_dot = arr[:, 19]

PLOT_TIME = 0
if PLOT_TIME:
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(t, ps2, 'k--')
    ax.set_ylabel(r'$p_{s2}$ [bar]', size = 12)
    ax.set_xlabel('t [s]', size=12)
    ax.set_title('Static pressure in state 2', size = 15)
    ax.grid()
    ax.set_xlim(0, t[-1])
    fig.savefig('images/ps2_vs_t.pdf', format='pdf')

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(t, ps3, 'k--', label='Static pressure')
    ax.plot(t, pt3, 'k-', label='Total pressure')
    ax.set_ylabel(r'pressure [bar]', size = 12)
    ax.set_xlabel('t [s]', size=12)
    ax.set_title('Static and total pressures in state 3', size = 15)
    ax.grid()
    ax.set_xlim(0, t[-1])
    ax.legend()
    fig.savefig('images/p3_vs_t.pdf', format='pdf')

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(t, pt4, 'k-')
    ax.set_ylabel(r'$p_{t4}$ [bar]', size = 12)
    ax.set_xlabel('t [s]', size=12)
    ax.set_title('Total pressure in state 4', size = 15)
    ax.grid()
    ax.set_xlim(0, t[-1])
    fig.savefig('images/pt4_vs_t.pdf', format='pdf')

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(t, pt5, 'k-')
    ax.set_ylabel(r'$p_{t5}$ [bar]', size = 12)
    ax.set_xlabel('t [s]', size=12)
    ax.set_title('Total pressure in state 5', size = 15)
    ax.grid()
    ax.set_xlim(0, t[-1])
    fig.savefig('images/pt5_vs_t.pdf', format='pdf')

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(t, Tt3, 'k-')
    ax.set_ylabel(r'$T_{t3}$ [°C]', size = 12)
    ax.set_xlabel('t [s]', size=12)
    ax.set_title('Total temperature in state 3', size = 15)
    ax.grid()
    ax.set_xlim(0, t[-1])
    fig.savefig('images/Tt3_vs_t.pdf', format='pdf')

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(t, Tt4, 'k-')
    ax.set_ylabel(r'$T_{t4}$ [°C]', size = 12)
    ax.set_xlabel('t [s]', size=12)
    ax.set_title('Total temperature in state 4', size = 15)
    ax.grid()
    ax.set_xlim(0, t[-1])
    fig.savefig('images/Tt4_vs_t.pdf', format='pdf')

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(t, Tt5, 'k-')
    ax.set_ylabel(r'$T_{t5}$ [°C]', size = 12)
    ax.set_xlabel('t [s]', size=12)
    ax.set_title('Total temperature in state 5', size = 15)
    ax.grid()
    ax.set_xlim(0, t[-1])
    fig.savefig('images/Tt5_vs_t.pdf', format='pdf')

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(t, Tt6, 'k-')
    ax.set_ylabel(r'$T_{t6}$ [°C]', size = 12)
    ax.set_xlabel('t [s]', size=12)
    ax.set_title('Total temperature in state 6', size = 15)
    ax.grid()
    ax.set_xlim(0, t[-1])
    fig.savefig('images/Tt6_vs_t.pdf', format='pdf')

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(t, Thrust - T_offset, 'k-')
    ax.set_ylabel(r'$T$ [N]', size = 12)
    ax.set_xlabel('t [s]', size=12)
    ax.set_title('Thrust (offset corrected)', size = 15)
    ax.grid()
    ax.set_xlim(0, t[-1])
    fig.savefig('images/Thrust_vs_t.pdf', format='pdf')

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(t, RPM, 'k-')
    ax.set_ylabel('RPM [rot/min]', size = 12)
    ax.set_xlabel('t [s]', size=12)
    ax.set_title('Rotational speed', size = 15)
    ax.grid()
    ax.set_xlim(0, t[-1])
    fig.savefig('images/RPM_vs_t.pdf', format='pdf')

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(t, m_dot - m_dot_offset, 'k-')
    ax.set_ylabel(r'$\dot{m}_f$ [g/s]', size = 12)
    ax.set_xlabel('t [s]', size=12)
    ax.set_title('Fuel mass ratio (offset corrected)', size = 15)
    ax.grid()
    ax.set_xlim(0, t[-1])
    fig.savefig('images/m_dot_vs_t.pdf', format='pdf')

    plt.show()

    plt.close('all')
