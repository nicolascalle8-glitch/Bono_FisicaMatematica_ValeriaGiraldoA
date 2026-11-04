"""
Bono Computacional Supletorio - Quiz 4 Física Matemática (Universidad EIA)

Evolución de un paquete gaussiano de partícula libre en una caja rígida [0, L]:

    du/dt = -alpha d2w/dx2 ,   dw/dt = alpha d2u/dx2 ,   Psi = u + i w
    u = w = 0 en x = 0 y x = L

Método: Crank-Nicolson (incondicionalmente estable y conserva la norma).

Genera en la carpeta 'figuras/':
  - densidad_probabilidad.png : rho(x,t) y u(x,t) en t = 0, 0.08, 0.14, 0.22 s
  - conservacion_norma.png    : P(t) a lo largo de la simulación

Uso:      python bono_schrodinger.py
Requiere: numpy, scipy, matplotlib
"""

import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")          # guarda las figuras sin necesidad de pantalla
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parámetros físicos (Tabla 1)
# ---------------------------------------------------------------------------
L = 1.0          # longitud de la cavidad [m]
alpha = 0.05     # coeficiente de dispersión hbar/(2m) [m^2/s]
x0 = 0.35        # posición inicial del paquete [m]
sigma = 0.07     # ancho inicial del paquete [m]
k0 = 50.0        # número de onda portador [1/m]
t_final = 0.25   # tiempo total de simulación [s]

# ---------------------------------------------------------------------------
# Parámetros numéricos
#   dx = 1 mm  -> k0*dx = 0.05 (unos 125 puntos por longitud de onda)
#   dt = 0.1 ms -> alpha*k0^2*dt = 0.0125 rad de fase por paso
#   r = alpha*dt/dx^2 = 5 (con FTCS esto explotaría; con CN no hay problema)
# ---------------------------------------------------------------------------
Nx = 1000                          # número de intervalos espaciales
dt = 1.0e-4                        # paso temporal [s]
T_PLOT = [0.0, 0.08, 0.14, 0.22]   # instantes que se grafican [s]

x = np.linspace(0.0, L, Nx + 1)    # nodos x_j = j*dx, j = 0..Nx
dx = x[1] - x[0]
r = alpha * dt / dx ** 2           # parámetro de malla (ec. 8)

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
os.makedirs(CARPETA, exist_ok=True)


def norma(u, w):
    """Probabilidad total discreta P = sum_j (u_j^2 + w_j^2) dx."""
    return np.sum(u ** 2 + w ** 2) * dx


# ---------------------------------------------------------------------------
# Condición inicial (ec. 4)
# ---------------------------------------------------------------------------
A = (sigma * np.sqrt(np.pi)) ** -0.5
envolvente = A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
u = envolvente * np.cos(k0 * x)
w = envolvente * np.sin(k0 * x)
u[0] = u[-1] = w[0] = w[-1] = 0.0          # fronteras de Dirichlet (ec. 3)

# ---------------------------------------------------------------------------
# Matrices de Crank-Nicolson
#
# Para los N = Nx-1 nodos interiores, D2 = tridiag(1, -2, 1). El esquema es
#     u^{n+1} + (r/2) D2 w^{n+1} = u^n - (r/2) D2 w^n
#     w^{n+1} - (r/2) D2 u^{n+1} = w^n + (r/2) D2 u^n
# que se escribe como un sistema por bloques  M_imp [u;w]^{n+1} = M_exp [u;w]^n.
# ---------------------------------------------------------------------------
N = Nx - 1
D2 = sp.diags([np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)], [-1, 0, 1])
I = sp.identity(N)
M_imp = sp.bmat([[I, (r / 2) * D2], [-(r / 2) * D2, I]], format="csc")
M_exp = sp.bmat([[I, -(r / 2) * D2], [(r / 2) * D2, I]], format="csc")
lu = spla.splu(M_imp)        # se factoriza una sola vez (la matriz no cambia)

# ---------------------------------------------------------------------------
# Integración en el tiempo
# ---------------------------------------------------------------------------
n_pasos = int(round(t_final / dt))
pasos_graficar = {int(round(t / dt)): t for t in T_PLOT}

tiempos = [0.0]
P = [norma(u, w)]
fotos = {0.0: (u.copy(), w.copy())}

y = np.concatenate([u[1:-1], w[1:-1]])   # vector de incógnitas [u; w]
for n in range(1, n_pasos + 1):
    y = lu.solve(M_exp @ y)              # un paso de Crank-Nicolson
    u[1:-1], w[1:-1] = y[:N], y[N:]      # las fronteras siguen en cero
    tiempos.append(n * dt)
    P.append(norma(u, w))
    if n in pasos_graficar:
        fotos[pasos_graficar[n]] = (u.copy(), w.copy())

tiempos, P = np.array(tiempos), np.array(P)
print(f"dx = {dx:.1e} m, dt = {dt:.1e} s, r = {r:.2f}")
print(f"P(0) = {P[0]:.12f}")
print(f"max |P(t) - P(0)| = {np.max(np.abs(P - P[0])):.2e}")

# ---------------------------------------------------------------------------
# Figura 1: densidad de probabilidad y parte real en 4 instantes
# ---------------------------------------------------------------------------
fig, ejes = plt.subplots(len(T_PLOT), 1, figsize=(8, 10), sharex=True)
for ax, t in zip(ejes, T_PLOT):
    uu, ww = fotos[t]
    ax.plot(x, uu, color="tab:blue", lw=0.9, alpha=0.75,
            label=r"$u(x,t)$  [m$^{-1/2}$]")
    ax.plot(x, uu ** 2 + ww ** 2, color="tab:red", lw=2,
            label=r"$\rho(x,t)=u^2+w^2$  [m$^{-1}$]")
    ax.set_title(f"t = {t:.2f} s", loc="left")
    ax.grid(alpha=0.3)
ejes[0].legend(loc="upper right", fontsize=9)
ejes[-1].set_xlabel("x [m]")
fig.suptitle("Paquete gaussiano en una caja rígida (Crank-Nicolson)")
fig.tight_layout()
fig.savefig(os.path.join(CARPETA, "densidad_probabilidad.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura 2: conservación de la norma P(t)
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
a1.plot(tiempos, P, color="tab:green")
a1.set_ylim(0.99, 1.01)
a1.set_ylabel("P(t)")
a1.set_title("Conservación de la probabilidad total")
a1.grid(alpha=0.3)
a2.plot(tiempos, P - P[0], color="tab:purple")
a2.set_ylabel(r"$P(t)-P(0)$")
a2.set_xlabel("t [s]")
a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(CARPETA, "conservacion_norma.png"), dpi=150)
plt.close(fig)

print(f"Figuras guardadas en: {CARPETA}")
