import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def readTable(path):
    vals=[]
    nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]
    with open(path) as f:
        for line in f.readlines():
            temp = [line.split(": ")[i].split(",")[0] for i in [1,3,4,5,6,7,8,9]]
            temp[-1] = temp[-1][:-1]
            vals.append(temp)
    table=np.array(vals, dtype=float)
    #  ~table[:,2]=np.round(np.exp(table[:,2]))
    #  ~table[:,4]=np.exp(table[:,4])
    #  ~table[:,5]=np.exp(table[:,5])
    table[:,6]=np.round(table[:,6])
    table[:,7]=[nodeFacs[int(i)] for i in (np.round(table[:,7]))]
    return table

def readTableNoLamb(path):
    vals=[]
    nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]
    with open(path) as f:
        for line in f.readlines():
            temp = [line.split(": ")[i].split(",")[0] for i in [1,3,4,5,6,7,8]]
            temp[-1] = temp[-1][:-1]
            vals.append(temp)
    table=np.array(vals, dtype=float)
    #  ~table[:,2]=np.round(np.exp(table[:,2]))
    #  ~table[:,4]=np.exp(table[:,4])
    table[:,5]=np.round(table[:,5])
    table[:,6]=[nodeFacs[int(i)] for i in (np.round(table[:,6]))]
    return table

def prntTable(table):
    struct=r"\begin{tabular}{"
    for i in range(len(table[0])):
        struct+="c|"
    struct=struct+"c}"
    print(struct)
    for j,tab in enumerate(table):
        string=r"    {} & ".format(int(j+1))
        for i,val in enumerate(tab):
            if i==2: 
                string+="{} & ".format(int(val))
            elif i==0: 
                string+="{:.4g} & ".format(val)
            else:
                string+="{:.3g} & ".format(val)
        string=string[:-2]+r"\\"
        print(string)

def plotDF(df):
    #  ~fig = px.parallel_coordinates(df, color="val_loss", labels={"val_loss": "validation loss", "alpha": "alpha", "batch_size": "batch size", "dropout": "dropout", "lambda": "lambda", "learn_rate": "learn rate", "n_layers": "n_layers", "n_nodes": "nodes"}, color_continuous_scale=px.colors.sequential.Bluered)
    fig = px.parallel_coordinates(df, color="val_loss", labels={"val_loss": "validation loss", "alpha": "alpha", "batch_size": "batch size", "dropout": "dropout", "learn_rate": "learn rate", "n_layers": "n_layers", "n_nodes": "nodes"}, color_continuous_scale=px.colors.sequential.Bluered)
    fig.show()
    
def plotDFv2noLamb(df):
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = df["val_loss"],
                       colorscale = "Bluered",
                       showscale = True,
                       cmin = np.min(df["val_loss"]),
                       cmax = np.min(df["val_loss"])+3),
            dimensions = list([
                dict(label = "validation loss", values = df['val_loss'],
                    range=(np.min(df["val_loss"]), np.min(df["val_loss"])+6)),
                dict(label = "alpha", values = df['alpha']),
                dict(label = "batch size", values = df['batch_size'],
                    tickvals = np.log(np.array([250, 500, 1000, 2000, 5000, 10000])),
                    ticktext = np.array([250, 500, 1000, 2000, 5000, 10000], dtype=str)),
                dict(label = "dropout", values = df['dropout']),
                dict(label = "learn rate", values = df['learn_rate'],
                    tickvals = np.log(np.array([0.00001, 0.0001, 0.001, 0.01, 0.05])),
                    ticktext = np.array([0.00001, 0.0001, 0.001, 0.01, 0.05], dtype=str)),
                dict(label = "n_layers", values = df['n_layers']),
                dict(label = "n_nodes", values = df['n_nodes'])])
        )
    )
    fig.show()
    
def plotDFv2(df):
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = df["val_loss"],
                       colorscale = "Plasma",
                       showscale = True,
                       cmin = np.min(df["val_loss"]),
                       cmax = np.min(df["val_loss"])+3),
            dimensions = list([
                dict(label = "validation loss", values = df['val_loss'],
                    range=(np.min(df["val_loss"]), np.min(df["val_loss"])+6)),
                dict(label = "alpha", values = df['alpha']),
                dict(label = "batch size", values = df['batch_size'],
                    tickvals = np.log(np.array([250, 500, 1000, 2000, 5000, 10000])),
                    ticktext = np.array([250, 500, 1000, 2000, 5000, 10000], dtype=str)),
                dict(label = "dropout", values = df['dropout']),
                dict(label = "lambda", values = df['lambda'],
                    tickvals = np.log(np.array([0.0001, 0.001, 0.01, 0.1])),
                    ticktext = np.array([0.0001, 0.001, 0.01, 0.1], dtype=str)),
                dict(label = "learn rate", values = df['learn_rate'],
                    tickvals = np.log(np.array([0.00001, 0.0001, 0.001, 0.01, 0.05])),
                    ticktext = np.array([0.00001, 0.0001, 0.001, 0.01, 0.05], dtype=str)),
                dict(label = "n_layers", values = df['n_layers']),
                dict(label = "n_nodes", values = df['n_nodes'])])
        )
    )
    fig.show()

def plotLoss(tabs, jname):
    plt.figure()
    colors=["blue", "red", "green"]
    labels=["logcosh", "MSE", "with early stopping"]
    plt.grid()
    #  ~for i, tab in enumerate(tabs):
        #  ~plt.scatter(np.arange(1,len(tab)+1,1), tab, color=colors[i], alpha=0.7, s=10, label=labels[i])
    plt.scatter(np.arange(1,len(tabs[0])+1,1), tabs[0], color=colors[0], alpha=0.7, s=10, label=labels[0])
    #  ~plt.scatter(np.arange(101,100+len(tabs[1][100:])+1,1), tabs[1][100:], color=colors[1], alpha=0.7, s=10, label=labels[1])
    #  ~plt.legend()
    plt.xlabel("Optimization Step")
    plt.ylabel(r"$-J_{\rm logcosh}^{\rm val}$")
    plt.tight_layout()
    #  ~plt.show()
    #  ~plt.savefig("lossplot"+jname[-10:-5]+".pdf")
    plt.savefig("lossOptimization.pdf")
    
def readFluct(path):
    vals = []
    feats = []
    with open(path) as f:
        for line in f.readlines():
            temp = [i for i in line.split(", ")[0:4]]
            feats = line.split(", ")[-1]
            vals.append(temp)
    table=np.array(vals, dtype=float)
    feats = [float(i) for i in feats.split(",")[1:]]
    print("Fluctuations for DNN with:\nlearn rate: {:.2g}, dropout: {:.2g}, lambda: {:.2g}, batch: {:.2g}, number of Layers: {:.2g}, node factor: {:.2g}, alpha: {:.2g}".format(*feats))
    print(r"\begin{tabular}{c|c|c|c|c}")
    print(r"    Target & Mean Value & Relative Deviation & min Value & max Value \\")
    nVar = ["Loss", "Logcosh", "Resolution Mean", "Resolution Std"]
    for j,tab in enumerate(table.T):
        #  ~print(tab)
        string = "    {} & ${:.5g} \pm {:.3g}$ & {:.2g}\% & {:.5g} & {:.5g}".format(nVar[j], np.mean(tab), np.std(tab), np.std(tab)/np.mean(tab), np.min(tab), np.max(tab))
        string += r" \\"
        print(string)
        #  ~print(len(tab))
        #  ~print(string
    print(r"\end{tabular}")

def plotYearCompBayes(filePath):
    nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]
    #  ~tabs = []
    meanVals = []
    offSet = 0.05
    fig, ax = plt.subplots(1,1)
    for j,year in enumerate(["2018", "2017", "2016_postVFP", "2016_preVFP"]):
        tab = readTable("/home/home4/institut_1b/nattland/DNN_ttbar/BayesOptLogs/"+year+filePath)
        #  ~print(tab[5:10,0])
        #  ~print(tab[:,0])
        means = -1*np.array([np.mean(tab[i:i+5,0]) for i in [5,0,10,15]])
        stds = np.array([np.std(tab[i:i+5,0]) for i in [5,0,10,15]])
        errs = stds/np.sqrt(5)
        meanVals.append(means)      
        if j==0:
            print(means, stds)
            print(r"DNN & learn rate & dropout & lambda & batch & number of Layers & node factor & alpha \\")
            tab[:,2]=np.round(np.exp(tab[:,2]))
            tab[:,4]=np.exp(tab[:,4])
            tab[:,5]=np.exp(tab[:,5])
            tab[:,6]=np.round(tab[:,6])
            for k in [0,5,10,15]: print("DNN 1 & {:.2g} & {:.4g} & {:.4g} & {:.2g} & {:.2g} & {:.2g} & {:.3g} \\\\".format(*tab[k,1:]))
            #  ~print(tab[k,:]
        #  ~ax.errorbar(np.linspace(0.85,3.85,4)+0.02*j, means, yerr=stds, label="std for"+year, linestyle="dashed", fmt='o', markersize=8, capsize=20, alpha=0.6)
        ax.errorbar(np.linspace(1-1.5*offSet,4-1.5*offSet,4)+offSet*j, means, yerr=errs, label=year.replace("_", " "), fmt='o', markersize=3, capsize=10, linestyle="dashed", alpha=0.8)
    for arr in meanVals:
        print("{:.4g} & {:.4g} & {:.4g} & {:.4g}".format(*arr))
    plt.xticks(ticks=[1,2,3,4], labels=["DNN 1", "DNN 2", "DNN 3", "DNN 4"])
    ax.set_ylabel(r"$J_{\rm logcosh}^{\rm val}$")
    ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
    ax.text(0.,1.,r"           $\,$Simulation$\,\bullet\,$Private work",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', style="italic", fontsize=10, color=(0.3,0.3,0.3))
    #  ~ax.text(1.,1.,r"$137.6\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig("yearComp.pdf")


if __name__ == "__main__":
    plotDF = False
    #  ~jname = "/home/home4/institut_1b/nattland/DNN_ttbar/BayesOptLogs/2018/Run7/fluctComp_v2.txt"
    jname = "/home/home4/institut_1b/nattland/DNN_ttbar/BayesOptLogs/2018/Run4/logBayes_V1.json"
    #  ~jname = "/home/home4/institut_1b/nattland/DNN_ttbar/BayesOptLogs/2018/Run4/Fluct_test_std.txt"
    
    #  ~readFluct(jname)
    plotYearCompBayes("/Run6/logBayes_yearComp.json")
    
    
    
    if plotDF:
        table = readTable(jname)
        print(table[np.argmax(table[:,0]),:])

        
        lossDF = pd.DataFrame({"val_loss": -1*table[:,0], "alpha": table[:,1], "batch_size": table[:,2], "dropout": table[:,3], "lambda": table[:,4], "learn_rate": table[:,5], "n_layers": table[:,6], "n_nodes": table[:,7]})
        #  ~lossDF = pd.DataFrame({"val_loss": -1*table[:,0], "alpha": table[:,1], "batch_size": table[:,2], "dropout": table[:,3], "learn_rate": table[:,4], "n_layers": table[:,5], "n_nodes": table[:,6]})
        #  ~print(lossDF.shape)
        plotDF(lossDF)
        #  ~plotDFv2(lossDF)
        #  ~plotDFv2(lossDF)
        #  ~plotLoss([table[:,0]], jname)
    
