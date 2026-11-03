"""
Bono Computacional Supletorio - Quiz 4 Física Matemática (Universidad EIA)

Paquete de ondas gaussiano de partícula libre en una caja rígida [0, L]:

    i dPsi/dt = -alpha d^2Psi/dx^2 ,  Psi = u + i w ,  Psi(0,t) = Psi(L,t) = 0

Este script:
  1. Muestra numéricamente que el esquema FTCS (Euler explícito) es inestable:
     la norma P crece sin control para cualquier r > 0.
  2. Implementa Crank-Nicolson (estable incondicionalmente y unitario).
  3. Valida Crank-Nicolson contra la solución exacta por el método de imágenes.
  4. Genera las figuras pedidas en la carpeta 'figuras/'.

Uso:  python simulacion_schrodinger.py
Requiere: numpy, scipy, matplotlib
"""

import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")          # permite guardar figuras sin pantalla
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parámetros (Tabla 1 del enunciado)
# ---------------------------------------------------------------------------
L = 1.0          # longitud de la cavidad [m]
alpha = 0.05     # coeficiente de dispersión hbar/(2m) [m^2/s]
x0 = 0.35        # posición inicial del paquete [m]
sigma = 0.07     # ancho inicial [m]
k0 = 50.0        # número de onda portador [1/m]
t_final = 0.25   # tiempo total [s]

# Malla numérica para Crank-Nicolson
Nx = 1000                # número de intervalos espaciales -> dx = 1 mm
dt = 1.0e-4              # paso temporal [s]
T_PLOT = [0.0, 0.08, 0.14, 0.22]   # instantes pedidos para la gráfica

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
os.makedirs(CARPETA, exist_ok=True)


# ---------------------------------------------------------------------------
# Herramientas comunes
# ---------------------------------------------------------------------------
def malla(Nx):
    """Devuelve x_j = j dx (j = 0..Nx) y dx."""
    x = np.linspace(0.0, L, Nx + 1)
    return x, x[1] - x[0]


def condicion_inicial(x):
    """Paquete gaussiano (ec. 4) como número complejo Psi = u + i w.
    Se imponen las fronteras de Dirichlet en los extremos (la gaussiana allí
    vale ~1e-6, así que el cambio es despreciable)."""
    A = (sigma * np.sqrt(np.pi)) ** -0.5
    psi = A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x)
    psi[0] = psi[-1] = 0.0
    return psi


def norma(psi, dx):
    """Probabilidad total discreta P = sum_j (u_j^2 + w_j^2) dx."""
    return np.sum(np.abs(psi) ** 2) * dx


def laplaciano(N):
    """Matriz tridiagonal D2 = tridiag(1, -2, 1) de tamaño N (solo nodos
    interiores; las fronteras Psi=0 quedan incluidas implícitamente)."""
    unos = np.ones(N)
    return sp.diags([unos[:-1], -2 * unos, unos[:-1]], [-1, 0, 1], format="csc")


# ---------------------------------------------------------------------------
# 1) Esquema FTCS (Euler explícito) - ecs. (6) y (7)
# ---------------------------------------------------------------------------
def paso_ftcs(u, w, r):
    """Un paso de las ecuaciones (6)-(7) del enunciado, escritas en u y w.
    Ambas se actualizan con los valores del nivel n (por eso se copian)."""
    un, wn = u.copy(), w.copy()
    u[1:-1] = un[1:-1] - r * (wn[2:] - 2 * wn[1:-1] + wn[:-2])
    w[1:-1] = wn[1:-1] + r * (un[2:] - 2 * un[1:-1] + un[:-2])
    return u, w


def correr_ftcs(r, Nx=200, n_pasos=60000):
    """Corre FTCS y devuelve tiempos y norma P(t)."""
    x, dx = malla(Nx)
    dt_loc = r * dx ** 2 / alpha
    psi = condicion_inicial(x)
    u, w = psi.real.copy(), psi.imag.copy()
    t, P = [0.0], [np.sum(u ** 2 + w ** 2) * dx]
    for n in range(1, n_pasos + 1):
        u, w = paso_ftcs(u, w, r)
        Pn = np.sum(u ** 2 + w ** 2) * dx
        t.append(n * dt_loc)
        P.append(Pn)
        if not np.isfinite(Pn) or Pn > 1e30:   # se detiene si explota
            break
    return np.array(t), np.array(P)


# ---------------------------------------------------------------------------
# 2) Esquema de Crank-Nicolson
#    (I - i r/2 D2) Psi^{n+1} = (I + i r/2 D2) Psi^n
#    En variables reales equivale a:
#      u^{n+1} + (r/2) D2 w^{n+1} = u^n - (r/2) D2 w^n
#      w^{n+1} - (r/2) D2 u^{n+1} = w^n + (r/2) D2 u^n
# ---------------------------------------------------------------------------
def crank_nicolson(Nx, dt, t_final, t_guardar):
    x, dx = malla(Nx)
    r = alpha * dt / dx ** 2
    N = Nx - 1                                # nodos interiores
    D2 = laplaciano(N)
    I = sp.identity(N, format="csc", dtype=complex)
    A = (I - 0.5j * r * D2).tocsc()           # matriz implícita (lado n+1)
    B = (I + 0.5j * r * D2).tocsc()           # matriz explícita (lado n)
    lu = spla.splu(A)                         # factorización LU una sola vez

    psi = condicion_inicial(x)
    n_pasos = int(round(t_final / dt))
    idx_guardar = {int(round(tg / dt)): tg for tg in t_guardar}

    tiempos, P = [0.0], [norma(psi, dx)]
    fotos = {}
    if 0 in idx_guardar:
        fotos[idx_guardar[0]] = psi.copy()
    for n in range(1, n_pasos + 1):
        psi[1:-1] = lu.solve(B @ psi[1:-1])   # avanza un paso
        tiempos.append(n * dt)
        P.append(norma(psi, dx))
        if n in idx_guardar:
            fotos[idx_guardar[n]] = psi.copy()
    return x, r, np.array(tiempos), np.array(P), fotos


# ---------------------------------------------------------------------------
# 3) Solución exacta (método de imágenes) para validar
#    Gaussiana libre en la recta real:
#      psi_libre = A e^{i(k x - alpha k^2 t)} (1+4 i alpha a t)^{-1/2}
#                  exp[-a (x - c - 2 alpha k t)^2 / (1 + 4 i alpha a t)],
#      a = 1/(2 sigma^2).
#    En la caja: extensión impar y 2L-periódica de la condición inicial.
# ---------------------------------------------------------------------------
def gaussiana_libre(x, t, c, k):
    A = (sigma * np.sqrt(np.pi)) ** -0.5
    a = 1.0 / (2 * sigma ** 2)
    z = 1 + 4j * alpha * a * t
    return (A / np.sqrt(z) * np.exp(1j * (k * x - alpha * k ** 2 * t))
            * np.exp(-a * (x - c - 2 * alpha * k * t) ** 2 / z))


def solucion_exacta_caja(x, t, n_img=4):
    psi = np.zeros_like(x, dtype=complex)
    for n in range(-n_img, n_img + 1):
        s = 2 * n * L
        # paquete original desplazado 2nL
        psi += gaussiana_libre(x - s, t, x0, k0)
        # imagen especular: g(-x) = A e^{-a(x+x0)^2} e^{-i k0 x}
        psi -= gaussiana_libre(x - s, t, -x0, -k0)
    return psi


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    # ---------------- Parte 1: factor de amplificación ----------------
    theta = np.linspace(0, np.pi, 400)          # theta = k dx
    plt.figure(figsize=(6.5, 4))
    for r in [0.05, 0.25, 0.5, 1.0]:
        s = 4 * r * np.sin(theta / 2) ** 2
        plt.plot(theta, np.sqrt(1 + s ** 2), label=f"FTCS, r = {r}")
    plt.plot(theta, np.ones_like(theta), "k--", lw=2,
             label="Crank-Nicolson (cualquier r)")
    plt.xlabel(r"$k\,\Delta x$")
    plt.ylabel(r"$|g(k)|$")
    plt.title("Factor de amplificación")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA, "1_factor_amplificacion.png"), dpi=150)
    plt.close()

    # ---------------- Parte 1: FTCS explota ----------------
    plt.figure(figsize=(6.5, 4))
    print("=== FTCS (Euler explícito) ===")
    for r in [0.01, 0.05, 0.1, 0.5]:
        t, P = correr_ftcs(r)
        print(f"r = {r:5.2f}: P inicial = {P[0]:.6f}, P final = {P[-1]:.3e} "
              f"tras {len(P)-1} pasos")
        plt.loglog(np.arange(1, len(P)+1), P, label=f"r = {r}")
    plt.xlabel("número de pasos n (escala log)")
    plt.ylabel(r"$P^n=\sum_j |\Psi_j^n|^2\Delta x$")
    plt.title("FTCS: la probabilidad total crece sin control")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA, "2_ftcs_norma.png"), dpi=150)
    plt.close()

    # ---------------- Parte 2: Crank-Nicolson ----------------
    t_comparar = [0.04, 0.14, 0.25]
    x, r, tiempos, P, fotos = crank_nicolson(Nx, dt, t_final,
                                             T_PLOT + t_comparar)
    dx = x[1] - x[0]
    print("\n=== Crank-Nicolson ===")
    print(f"dx = {dx:.1e} m, dt = {dt:.1e} s, r = {r:.2f}, "
          f"k0*dx = {k0*dx:.3f}")
    print(f"P(0) = {P[0]:.12f}")
    print(f"max |P(t) - P(0)| = {np.max(np.abs(P - P[0])):.2e}")

    # Validación contra la solución exacta (método de imágenes)
    print("\nError vs. solución exacta (imágenes):")
    for tc in t_comparar:
        ex = solucion_exacta_caja(x, tc)
        err = np.max(np.abs(np.abs(fotos[tc]) ** 2 - np.abs(ex) ** 2))
        print(f"  t = {tc:.2f} s: max|rho_num - rho_exacta| = {err:.2e}  "
              f"(max rho = {np.max(np.abs(ex)**2):.2f})")

    # Convergencia: refinando dt y dx el error baja ~4 veces (orden 2)
    print("\nConvergencia en t = 0.25 s (error máximo en rho):")
    for (Nx_c, dt_c) in [(500, 2e-4), (1000, 1e-4), (2000, 5e-5)]:
        xc, _, _, _, fc = crank_nicolson(Nx_c, dt_c, t_final, [t_final])
        e = np.max(np.abs(np.abs(fc[t_final]) ** 2
                          - np.abs(solucion_exacta_caja(xc, t_final)) ** 2))
        print(f"  Nx = {Nx_c:5d}, dt = {dt_c:.0e}: error = {e:.3e}")

    # Figura: densidad de probabilidad y parte real en 4 instantes
    fig, ejes = plt.subplots(len(T_PLOT), 1, figsize=(8, 10), sharex=True)
    for ax, tp in zip(ejes, T_PLOT):
        psi = fotos[tp]
        ex = solucion_exacta_caja(x, tp)
        ax.plot(x, psi.real, color="tab:blue", lw=0.8, alpha=0.7,
                label=r"$u(x,t)=\mathrm{Re}\,\Psi$")
        ax.plot(x, np.abs(psi) ** 2, color="tab:red", lw=2,
                label=r"$\rho(x,t)=u^2+w^2$ (CN)")
        ax.plot(x, np.abs(ex) ** 2, "k:", lw=1.2, label=r"$\rho$ exacta")
        ax.set_ylabel(r"[m$^{-1/2}$] / [m$^{-1}$]")
        ax.set_title(f"t = {tp:.2f} s", loc="left")
        ax.grid(alpha=0.3)
    ejes[0].legend(loc="upper right", fontsize=8)
    ejes[-1].set_xlabel("x [m]")
    fig.suptitle("Paquete gaussiano en caja rígida - Crank-Nicolson")
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA, "3_densidad_probabilidad.png"), dpi=150)
    plt.close(fig)

    # Figura: conservación de la norma
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    a1.plot(tiempos, P, color="tab:green")
    a1.set_ylim(0.99, 1.01)
    a1.set_ylabel("P(t)")
    a1.set_title("Conservación de la probabilidad total (Crank-Nicolson)")
    a1.grid(alpha=0.3)
    a2.plot(tiempos, P - P[0], color="tab:purple")
    a2.set_ylabel(r"$P(t)-P(0)$")
    a2.set_xlabel("t [s]")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA, "4_conservacion_norma.png"), dpi=150)
    plt.close(fig)

    # Mapa espacio-tiempo de rho (opcional, ayuda a ver la reflexión)
    _, _, _, _, f_all = crank_nicolson(Nx, dt, t_final,
                                       list(np.round(np.arange(0, 0.2501, 0.0025), 4)))
    ts = sorted(f_all)
    R = np.array([np.abs(f_all[tt]) ** 2 for tt in ts])
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(x, ts, R, shading="auto", cmap="inferno")
    plt.colorbar(label=r"$\rho(x,t)$ [m$^{-1}$]")
    plt.xlabel("x [m]")
    plt.ylabel("t [s]")
    plt.title("Densidad de probabilidad: choque con la pared x = L")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA, "5_mapa_xt.png"), dpi=150)
    plt.close()

    print(f"\nFiguras guardadas en: {CARPETA}")


if __name__ == "__main__":
    main()
