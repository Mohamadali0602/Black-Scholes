import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter

import scipy.stats as st

def SolutionAn(N, M, mu=1,tf=10 , L=200, r=0.06,sig=0.3,E=100):
    T=tf
    k=2*r/(sig*sig)
    t= np.linspace(0,T,N+1)
    #x=np.linspace(0,L,M)
    S=np.linspace(0,L,M)
    N= lambda y : st.norm.cdf(y , loc = 0, scale = 1) #fonction de répartition de la loi normale


    r1=(1/(sig*np.sqrt(T-t[:, None])))*np.log(S/E) - (k+1)*sig*np.sqrt(T-t[:, None])
    r2=(1/(sig*np.sqrt(T-t[:, None])))*np.log(S/E) - (k-1)*sig*np.sqrt(T-t[:, None])

    C = (S/E)*N(r1) - (1/2)*(np.exp(-r*(T-t[:, None])))*N(r2)

    return t/T, S, C

# Résolution avec les paramètres donnés
sig=0.3
E=100
r=0.06
t, S, C = SolutionAn(N=100,M=101,mu=1,tf=1,L=200,r=r,sig=sig,E=E)

# Affichage en 3D
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))

# Générer les grilles pour l'affichage
tt, SS = np.meshgrid(t, S)

# Création de la surface
surf = ax.plot_surface(tt.T, SS.T, C, cmap=cm.coolwarm,  # Notez le `.T` pour aligner les dimensions
                       linewidth=0, antialiased=False)
wire = ax.plot_wireframe(tt.T, SS.T, C, color='blue', linewidth=0.5, alpha=0.5)
# Ajuster les limites pour aligner les origines
ax.set_xlim(1, 0)  # t est normalisé entre 0 et 1
ax.set_ylim(0,200)  # S est défini de 0 à L
ax.set_zlim(-0.2, 1.8)
# Formater l'axe Z avec deux décimales
ax.zaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

# Ajout d'une barre de couleur
fig.colorbar(surf, shrink=0.5, aspect=10, label="C(S,t)")

# Labels des axes
ax.set_xlabel("Temps (t/T)")
ax.set_ylabel(" Prix du stock S")
ax.set_zlabel(" C ")

# Titre
ax.set_title(f"Graphique de la solution analytique de Black-Scholes pour $\sigma$ = {sig}, E = {E} et r = {100*r}% ")

# Afficher
plt.show()