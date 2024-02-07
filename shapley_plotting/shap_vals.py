import numpy as np
import matplotlib.pyplot as plt

var=[]

with open("ShapVals.txt") as f:
    temp = f.read()[2:].split("]\n[")
    for t in temp:
        t=t.replace("\n", " ")
        var.append(np.array(t.split("  "), dtype=float))

var2 = var[0]+var[1]
shVars = np.vstack((var[0], var[1], var2)).T
shVars = shVars[shVars[:, 2].argsort()][::-1]

#show only reduced set of 20 instead of all 54 input features
shVals = []
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,21,22]:
    shVals.append(shVars[i,:])
shVars = np.vstack(shVals)

logcoshvals = [16.828, 16.861, 16.823, 17.276] #logcosh as function of n_param=40,30,20,10
n_param = np.array([10, 20, 30, 40], dtype=str)
mean_vals = [16.801, 16.78, 16.794, 16.896, 16.913, 16.861] #logcosh fluctuations for errorbar

shapLabels = [r"$\rm PUPPI\;p_{x}^{\rm miss}$",
 r"$\rm PUPPI\;p_{y}^{\rm miss}$",
 r"$\rm PF\;p_{x}^{\rm miss}$",
 r"$\rm PF\;p_{y}^{\rm miss}$",
 r"$\rm p_{x}^{all\,j}$",
 r"$\rm p_{y}^{all\,j}$",
 r"$\rm m_{ll+all\,j}$",
 r"$\rm p_{x}^{j_{1}}$",
 r"$\rm m_{all\,j}$",
 r"$\rm p_{x}^{l_{1}}$",
 r"$\rm p_{y}^{l_{1}}$",
 r"$\rm p_{y}^{j_{1}}$",
 r"$\rm Calo\;p_{\rm T}^{\rm miss}$",
 r"$\rm M_{\rm T2}$",
 r"$\rm m_{jj}$",
 r"$\rm N_{\rm jets}$",
 r"$\rm E^{j_{1}}$",
 r"$\rm H_{\rm T}$",
 r"$\rm p_{x}^{j_{2}}$",
 r"$\rm p_{y}^{j_{2}}$"]


fig, ax = plt.subplots(1,1, figsize=(5,3.75))
ax.errorbar([1,2,3,4], logcoshvals[::-1], yerr=np.std(mean_vals), capsize=3, color="red")
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(n_param)
ax.set_ylabel(r"$J^{\,\rm val}_{\,\rm logcosh}(\theta)$")
ax.set_xlabel(r"number of input features $n_{\rm feat}$")
ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
ax.text(0.,1.,r"           $\,$Simulation$\,\bullet\,$Private work",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', style="italic", fontsize=10, color=(0.3,0.3,0.3))
ax.text(1.,1.,r"$59.8\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)
plt.tight_layout(pad=0.12)


fig, ax = plt.subplots(1,1, figsize=(6.5,5))
ax.barh(np.linspace(0,19,20),shVars[:20,1][::-1], tick_label=shapLabels[::-1],color="steelblue", alpha=0.8, label=r"$\rm PUPPI\;p_{x}^{\rm miss}-\rm gen. p_{x}^{\rm miss}}$")
ax.barh(np.linspace(0,19,20),shVars[:20,0][::-1], tick_label=shapLabels[::-1], left=shVars[:20,1][::-1], color="skyblue", alpha=0.8, label=r"$\rm PUPPI\;p_{y}^{\rm miss}-\rm gen. p_{y}^{\rm miss}}$")

ax.set_ylim(-0.7, 19.7)
ax.set_xlabel(r"Shapley value",fontsize=14)
ax.set_ylabel(r"Input feature",fontsize=14)
ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
ax.text(0.,1.,r"           $\,$Simulation$\,\bullet\,$Work in progress",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', style="italic", fontsize=9.5, color=(0.3,0.3,0.3))
ax.text(1.,1.,r"$59.8\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)
plt.legend()
plt.tight_layout(pad=0.12)
plt.savefig("plot.pdf")
