import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

def inicializar_parametros():
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

def inicializar_arreglos(n, nt, C0, Ri, Ro):
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
    t = np.linspace(0, nt * (5.0 / (nt - 1)), nt + 1)
    H = np.zeros((n, nt + 1))
    Cflux = np.zeros(n)
    Hflux = np.zeros((n, nt + 1))
    HDisp = np.zeros((n, nt + 1))
    HStress_r = np.zeros((n, nt + 1))
    HStress_t = np.zeros((n, nt + 1))
    HStrain_r = np.zeros((n, nt + 1))
    HStrain_t = np.zeros((n, nt + 1))
    
    Cold[:] = C0
    H[:, 0] = Cold
    Hflux[:, 0] = 0
    
    return Cold, C, Disp, sigma_r, sigma_t, epsi_r, epsi_t, A, ADisp, rhs, rhsDisp, r, t, H, Cflux, Hflux, HDisp, HStress_r, HStress_t, HStrain_r, HStrain_t

def configurar_matriz_difusion(matriz_difusion, matriz_dispersion, num_nodos, delta_r, difusion, factor_S, delta_tiempo, vector_radio, coef_nu):
    for i in range(1, num_nodos - 1):
        matriz_difusion[1 + i - i, i] = -2 * factor_S * difusion / delta_r**2 - 1 / delta_tiempo
        matriz_difusion[1 + i - (i + 1), i + 1] = factor_S * difusion / delta_r**2 + factor_S * difusion / (2 * vector_radio[i] * delta_r)
        matriz_difusion[1 + i - (i - 1), i - 1] = factor_S * difusion / delta_r**2 - factor_S * difusion / (2 * vector_radio[i] * delta_r)
        matriz_dispersion[2 + i - (i + 1), i + 1] = 1 / delta_r**2 + 1 / (2 * vector_radio[i] * delta_r)
        matriz_dispersion[2 + i - (i - 1), i - 1] = 1 / delta_r**2 - 1 / (2 * vector_radio[i] * delta_r)
        matriz_dispersion[2 + i - i, i] = -2 / delta_r**2 - 1.0 / vector_radio[i]**2
    
    matriz_difusion[1 + 0 - 0, 0] = 1 
    matriz_difusion[1 + num_nodos - 1 - (num_nodos - 1), num_nodos - 1] = 1
    matriz_dispersion[2 + 0 - 0, 0] = -3 * (coef_nu - 1.) / (2 * delta_r) - coef_nu / vector_radio[0]
    matriz_dispersion[2 + 0 - 1, 1] = 4 * (coef_nu - 1.) / (2 * delta_r)
    matriz_dispersion[2 + 0 - 2, 2] = -(coef_nu - 1.) / (2 * delta_r)
    matriz_dispersion[2 + num_nodos - 1 - (num_nodos - 3), num_nodos - 3] = (coef_nu - 1.) / (2 * delta_r)
    matriz_dispersion[2 + num_nodos - 1 - (num_nodos - 2), num_nodos - 2] = -4 * (coef_nu - 1.) / (2 * delta_r)
    matriz_dispersion[2 + num_nodos - 1 - (num_nodos - 1), num_nodos - 1] = 3 * (coef_nu - 1.) / (2 * delta_r) - coef_nu / vector_radio[num_nodos - 1]

def resolver_difusion(matriz_difusion, vector_rhs, concentracion_old, delta_tiempo, num_tiempos, conc_inicial, conc_final, delta_r, difusion, vector_radio, factor_S, matriz_H, matriz_Hflux, num_nodos):
    for j in range(num_tiempos):
        for i in range(1, num_nodos - 1):
            vector_rhs[i] = -concentracion_old[i] / delta_tiempo
        vector_rhs[0] = conc_inicial
        vector_rhs[num_nodos - 1] = conc_final
        concentracion = solve_banded((1, 1), matriz_difusion, vector_rhs)
        
        # Post procesamiento de flujos
        for i in range(1, num_nodos - 1):
            matriz_Hflux[i, j + 1] = -2.0 * np.pi * vector_radio[i] * difusion * (concentracion[i + 1] - concentracion[i - 1]) / (2 * delta_r)
        matriz_Hflux[0, j + 1] = -2.0 * np.pi * vector_radio[0] * difusion * (-3 * concentracion[0] + 4 * concentracion[1] - concentracion[2]) / (2 * delta_r)  
        matriz_Hflux[num_nodos - 1, j + 1] = -2.0 * np.pi * vector_radio[num_nodos - 1] * difusion * (3 * concentracion[num_nodos - 1] - 4 * concentracion[num_nodos - 2] + concentracion[num_nodos - 3]) / (2 * delta_r)
        matriz_H[:, j + 1] = concentracion
        concentracion_old[:] = concentracion

def resolver_esfuerzos(matriz_dispersion, vector_rhs_disp, desplazamiento, matriz_H, matriz_HDisp, num_nodos, num_tiempos, delta_r, omega, presion_interna, modulo_E, coef_nu):
    for j in range(num_tiempos + 1):
        concentracion = matriz_H[:, j]
        for i in range(1, num_nodos - 1):
            vector_rhs_disp[i] = -(1.0 / 3.0) * (omega / (coef_nu - 1.0)) * (concentracion[i + 1] - concentracion[i - 1]) / (2.0 * delta_r)
        vector_rhs_disp[0] = -presion_interna * (1.0 + coef_nu) * (2 * coef_nu - 1.0) / modulo_E - (1.0 / 3.0) * omega * concentracion[0]
        vector_rhs_disp[num_nodos - 1] = -(1.0 / 3.0) * omega * concentracion[num_nodos - 1]
        desplazamiento = solve_banded((2, 2), matriz_dispersion, vector_rhs_disp)
        matriz_HDisp[:, j] = desplazamiento

def post_procesar_esfuerzos(matriz_HDisp, matriz_HStress_r, matriz_HStress_t, matriz_HStrain_r, matriz_HStrain_t, vector_radio, num_nodos, modulo_E, coef_nu, omega, matriz_H, delta_r):
    for j in range(matriz_HDisp.shape[1]):
        desplazamiento = matriz_HDisp[:, j]
        epsi_t = desplazamiento / vector_radio  # actualizamos epsi_t como deformación tangencial
        
        for i in range(num_nodos):
            # Calculamos la deformación radial epsi_r
            if i > 0 and i < num_nodos - 1:
                epsi_r = (desplazamiento[i + 1] - desplazamiento[i - 1]) / (2.0 * delta_r)
            elif i == 0:
                epsi_r = (-3.0 * desplazamiento[0] + 4.0 * desplazamiento[1] - desplazamiento[2]) / (2.0 * delta_r)
            elif i == num_nodos - 1:
                epsi_r = (3.0 * desplazamiento[num_nodos - 1] - 4.0 * desplazamiento[num_nodos - 2] + desplazamiento[num_nodos - 3]) / (2.0 * delta_r)
                
            sigma_r = modulo_E / (1.0 - coef_nu**2) * (epsi_r + coef_nu * epsi_t[i]) - omega * matriz_H[i, j]
            sigma_t = modulo_E / (1.0 - coef_nu**2) * (epsi_t[i] + coef_nu * epsi_r) - omega * matriz_H[i, j]
            
            matriz_HStress_r[i, j] = sigma_r
            matriz_HStress_t[i, j] = sigma_t
            matriz_HStrain_r[i, j] = epsi_r
            matriz_HStrain_t[i, j] = epsi_t[i]

#ejecución
Ro, Ri, Cin, Cout, D, E, nu, Omega, C0, pin, n, dr, S, t_end, nt, dt = inicializar_parametros()
Cold, C, Disp, sigma_r, sigma_t, epsi_r, epsi_t, A, ADisp, rhs, rhsDisp, r, t, H, Cflux, Hflux, HDisp, matriz_HStress_r, matriz_HStress_t, matriz_HStrain_r, matriz_HStrain_t = inicializar_arreglos(n, nt, C0, Ri, Ro)

configurar_matriz_difusion(A, ADisp, n, dr, D, S, dt, r, nu)
resolver_difusion(A, rhs, Cold, dt, nt, Cin, Cout, dr, D, r, S, H, Hflux, n)
resolver_esfuerzos(ADisp, rhsDisp, Disp, H, HDisp, n, nt, dr, Omega, pin, E, nu)
post_procesar_esfuerzos(HDisp, matriz_HStress_r, matriz_HStress_t, matriz_HStrain_r, matriz_HStrain_t, r, n, E, nu, Omega, H, dr)

# Gráfica sigma_r vs radio para cada tiempo
plt.figure()
for i in range(0, nt + 1, 100):
    plt.plot(r * 1e3, matriz_HStress_r[:, i], label=f't={i * dt:.1f} years') 
plt.xlabel('r [mm]')
plt.ylabel('$\sigma_r$ [Pa]') 
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Gráfica sigma_t vs radio para cada tiempo
plt.figure()
for i in range(0, nt + 1, 100):
    plt.plot(r * 1e3, matriz_HStress_t[:, i], label=f't={i * dt:.1f} years') 
plt.xlabel('r [mm]')
plt.ylabel(r'$\sigma_\theta$ [Pa]')  
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()


if __name__ == "__main__":
    main()
