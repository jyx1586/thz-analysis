from scipy.integrate import solve_ivp
import numpy as np
from math import log, sqrt, pi, exp
from matplotlib import pyplot as plt

U = 5.21e9 # J/m^3 energy density, absorbance = 0.03
tl = 60e-15 # laser pulse duration, 60 fs
S0 = U * sqrt(4*log(2)/pi) / tl # max power of S(t)
t0 = 0 # time when laser pulse arrives, 500 fs
CE = 66 # electronic heat capacity coefficient, Jm^-3 K^-2
CP = 100*CE # phononic heat capacity coefficient (Cp = 100Ce)
G = 5e16 # electron-phonon interaction coefficient, Jm^-3 Sec^-1 K^-1


# functions
# power depositied per unit volume
def S(t):
    return S0*exp((-4)*log(2)*(((t-t0)/tl)**2))
# electronic heat capacity as a function of electron temperature
def Ce(Te): 
    return CE * Te

def ttm(t, y):
    Te, T = y # unpack state vector
    s = S(t)
    ce = Ce(Te)
    dTe_dt = ((-G) * (Te-T) + s)/ce
    dT_dt = (G * (Te-T))/(CP*ce)

    return [dTe_dt, dT_dt]


times = np.linspace(-2e-12, 10e-12, 300)
sol = solve_ivp(ttm, [-2e-12, 10e-12], [300, 300], t_eval=times, max_step=1e-15)
t = sol.t
Te = sol.y[0]
T = sol.y[1]
print()
print(Te.shape, T.shape)

plt.figure()
plt.plot(t, Te)
plt.plot(t, T)
plt.show()


