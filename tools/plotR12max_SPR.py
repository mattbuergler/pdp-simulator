import numpy as np
from numpy import linalg
import pandas as pd
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import sys
import pathlib

path = pathlib.Path(sys.argv[1])

SPRloop = np.asarray(pd.read_csv(path / 'SPR.csv',names=['SPR']).values)
Rxymaxloop = np.asarray(pd.read_csv(path / 'R12max.csv',names=['Rxymaxloop']).values)
uinstloop = np.asarray(pd.read_csv(path / 'U.csv',names=['U']).values)
spr_ineq = np.linspace(0.0,1.0,1000)
rmax_ineq = 0.4*(np.square(spr_ineq) + 1.0)
fig, ax = plt.subplots(1,1,figsize=(4,2.88))
sc = plt.scatter(SPRloop,Rxymaxloop,c=uinstloop,cmap='viridis')
plt.plot(spr_ineq,rmax_ineq,color='k')
ax.set_ylabel(r'$R_{\mathrm{12},i,\mathrm{max}}$ [-]')
ax.set_xlabel(r'SPR$_i$ [-]')
ax.set_xlim([0.0,1.0])
#ax.set_ylim([0.0,1.0])
ax.grid()
cbar = plt.colorbar(sc)
cbar.set_label(r'$U_x$ [m/s]', rotation=90)
plt.subplots_adjust(left=0.16, bottom=0.15, right=0.88, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path / 'SPR_R12max.svg',dpi=300)