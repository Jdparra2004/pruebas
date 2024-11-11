import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

def initialize_parameters():
    Ro = 2.0e-2
    Ri = Ro - 3.0e-3
    Cin = 0.1
    Cout = 0.0
    D = 1.0e-8
    E = 10e9  # Pa
    nu = 0.3
    Omega = 5.0e-3
    C0 = 0.0
    pin = 400.0e3  # Pa
    n = 2000  # nodes
    dr = (Ro - Ri) / (n - 1)
    S = 60 * 60  # seconds per hour
    t_end = 5.0
    nt = 1000
    dt = t_end / (nt - 1)
    return Ro, Ri, Cin, Cout, D, E, nu, Omega, C0, pin, n, dr, S, t_end, nt, dt

def initialize_arrays(n, nt, C0, Ri, Ro):
    Cold = np.zeros(n)
    C = np.zeros(n)
    Disp = np.zeros(n)
    sigma_r = np.zeros(n)
    sigma_t = np.zeros(n)
    epsi_r = np.zeros(n)
    epsi_t = np.zeros(n)
    A = np.zeros((3, n))
    ADisp = np.zeros((5, n))
    rhs = np.zeros(n)
    rhsDisp = np.zeros(n)
    r = np.linspace(Ri, Ro, n)
    t = np.linspace(0, nt * dt, nt + 1)
    H = np.zeros((n, nt + 1))
    Cflux = np.zeros((n, nt + 1)) # Array de 2D
    Hflux = np.zeros((n, nt + 1)) # Array de 2D
    HDisp = np.zeros((n, nt + 1))
    HStress_r = np.zeros((n, nt + 1))
    HStress_t = np.zeros((n, nt + 1))
    HStrain_r = np.zeros((n, nt + 1))
    HStrain_t = np.zeros((n, nt + 1))
    Cold[:] = C0
    H[:, 0] = Cold
    Hflux[:, 0] = 0
    return (Cold, C, Disp, sigma_r, sigma_t, epsi_r, epsi_t, A, ADisp, rhs, rhsDisp, r, t, H, Cflux, Hflux, HDisp, HStress_r, HStress_t, HStrain_r, HStrain_t)

def setup_diffusion_matrix(A, ADisp, n, dr, D, S, dt, r, nu):
    for i in range(1, n - 1):
        A[1 + i - i, i] = -2 * S * D / dr**2 - 1 / dt
        A[1 + i - (i + 1), i + 1] = S * D / dr**2 + S * D / (2 * r[i] * dr)
        A[1 + i - (i - 1), i - 1] = S * D / dr**2 - S * D / (2 * r[i] * dr)
        ADisp[2 + i - (i + 1), i + 1] = 1 / dr**2 + 1 / (2 * r[i] * dr)
        ADisp[2 + i - (i - 1), i - 1] = 1 / dr**2 - 1 / (2 * r[i] * dr)
        ADisp[2 + i - i, i] = -2 / dr**2 - 1.0 / r[i]**2
    A[1 + 0 - 0, 0] = 1
    A[1 + n - 1 - (n - 1), n - 1] = 1
    ADisp[2 + 0 - 0, 0] = -3 * (nu - 1.0) / (2 * dr) - nu / r[0]
    ADisp[2 + 0 - 1, 1] = 4 * (nu - 1.0) / (2 * dr)
    ADisp[2 + 0 - 2, 2] = -(nu - 1.0) / (2 * dr)
    ADisp[2 + n - 1 - (n - 3), n - 3] = (nu - 1.0) / (2 * dr)
    ADisp[2 + n - 1 - (n - 2), n - 2] = -4 * (nu - 1.0) / (2 * dr)
    ADisp[2 + n - 1 - (n - 1), n - 1] = 3 * (nu - 1.0) / (2 * dr) - nu / r[n - 1]
    return A, ADisp

def solve_diffusion(A, rhs, Cold, dt, nt, Cin, Cout, dr, D, r, S, H, Hflux, n):
    Cflux = np.zeros((n, nt + 1))  # Asegúrate de que Cflux sea 2D

    for j in range(nt):
        for i in range(1, n - 1):
            rhs[i] = -Cold[i] / dt
        rhs[0] = Cin
        rhs[n - 1] = Cout
        C = solve_banded((1, 1), A, rhs)
        
        # post-processing de flujos
        for i in range(1, n - 1):
            Cflux[i, j + 1] = -2.0 * np.pi * r[i] * D * (C[i + 1] - C[i - 1]) / (2 * dr)
        Cflux[0, j + 1] = -2.0 * np.pi * r[0] * D * (-3 * C[0] + 4 * C[1] - C[2]) / (2 * dr)  
        Cflux[n - 1, j + 1] = -2.0 * np.pi * r[n - 1] * D * (3 * C[n - 1] - 4 * C[n - 2] + C[n - 3]) / (2 * dr)
        
        H[:, j + 1] = C
        Hflux[:, j + 1] = Cflux[:, j + 1]  # Asegúrate de que esto sea correcto
        Cold[:] = C

def solve_disp(ADisp, rhsDisp, Disp, H, HDisp, n, nt, dt, dr, Omega, pin, E, nu):
    for j in range(nt + 1):
        C = H[:, j]
        for i in range(1, n - 1):
            rhsDisp[i] = -(1.0 / 3.0) * (Omega / (nu - 1.0)) * (C[i + 1] - C[i - 1]) / (2.0 * dr)
        rhsDisp[0] = -pin * (1.0 + nu) * (2.0 * nu - 1.0) / E - (1.0 / 3.0) * Omega * C[0]
        rhsDisp[n - 1] = -(1.0 / 3.0) * Omega * C[n - 1]
        Disp = solve_banded((2, 2), ADisp, rhsDisp)
        HDisp[:, j] = Disp

def post_processing(n, nt, r, H, Hflux, Disp, HDisp, HStress_r, HStress_t, HStrain_r, HStrain_t, dt):
    tot_flux_in = 0
    tot_flux_out = 0
    for i in range(nt):  # Cambiado a range(nt)
        tot_flux_in += 0.5 * (Hflux[0, i] + Hflux[0, i + 1]) * S * dt
        tot_flux_out += 0.5 * (Hflux[n - 1, i] + Hflux[n - 1, i + 1]) * S * dt
    
    for j in range(nt + 1):
        epsi_t = Disp / r
        for i in range(n):
            if i > 0 and i < n - 1:
                epsi_r[i] = (Disp[i + 1] - Disp[i - 1]) / (2 * dr)
            if i == 0:
                epsi_r[i] = (-3 * Disp[i] + 4 * Disp[i + 1] - Disp[i + 2]) / (2 * dr)
            if i == n - 1:
                epsi_r[i] = (3 * Disp[i] - 4 * Disp[i - 1] + Disp[i - 2]) / (2 * dr)
            sigma_r[i] = (E / ((1.0 + nu) * (2.0 * nu - 1.0))) * ((nu - 1.0) * epsi_r[i] - nu * epsi_t[i] + (1.0 / 3.0) * Omega * C[i])
            HStress_r[i] = sigma_r[i]
            HStress_t[i] = sigma_r[i] * (1 - nu)
            HStrain_r[i] = epsi_r[i]
            HStrain_t[i] = epsi_t[i]

# Inicializa los parámetros y las variables necesarias
Ro, Ri, Cin, Cout, D, E, nu, Omega, C0, pin, n, dr, S, t_end, nt, dt = initialize_parameters()

# Inicializa los arreglos
Cold, C, Disp, sigma_r, sigma_t, epsi_r, epsi_t, A, ADisp, rhs, rhsDisp, r, t, H, Cflux, Hflux, HDisp, HStress_r, HStress_t, HStrain_r, HStrain_t = initialize_arrays(n, nt, C0, Ri, Ro)

# Configura la matriz de difusión
A, ADisp = setup_diffusion_matrix(A, ADisp, n, dr, D, S, dt, r, nu)

# Resuelve la difusión
solve_diffusion(A, rhs, Cold, dt, nt, Cin, Cout, dr, D, r, S, H, Hflux, n)

# Resuelve el desplazamiento
solve_disp(ADisp, rhsDisp, Disp, H, HDisp, n, nt, dt, dr, Omega, pin, E, nu)

# Procesamiento posterior
post_processing(n, nt, r, H, Hflux, Disp, HDisp, HStress_r, HStress_t, HStrain_r, HStrain_t, dt)

#grafica sigma_r vs radio para cada tiempo
plt.figure()
for i in range(0,nt+1,100):
    plt.plot(r*1e3,HStress_r[:,i],label=f't={i*dt:.1f} years') 
plt.xlabel('r [mm]')
plt.ylabel('$\sigma_r$ [Pa]') 
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

#grafica sigma_t vs radio para cada tiempo
plt.figure()
for i in range(0,nt+1,100):
    plt.plot(r*1e3,HStress_t[:,i],label=f't={i*dt:.1f} years') 
plt.xlabel('r [mm]')
plt.ylabel(r'$\sigma_\theta$ [Pa]')  
#https://stackoverflow.com/questions/10370760/matplotlib-axis-label-theta-does-not-work-theta-does
#If you specify that the string is raw text (a r before the quotation mark), it works
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()

print("Forma de Hflux:", Hflux.shape)
print("Forma de Cflux:", Cflux.shape)

