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
    plt.scatter(np.arange(101,100+len(tabs[1][100:])+1,1), tabs[1][100:], color=colors[1], alpha=0.7, s=10, label=labels[1])
    plt.legend()
    plt.axvline(40, color="r")
    plt.axvline(100, color="r")
    plt.axvline(160, color="r")
    plt.xlabel("Iteration")
    plt.ylabel("-MSE")
    plt.tight_layout()
    #  ~plt.show()
    #  ~plt.savefig("lossplot"+jname[-10:-5]+".pdf")
    plt.savefig("lossComp_MSE_logc.pdf")


if __name__ == "__main__":
    #  ~table = readTable("HParaLogs/logBayes_C2_3.json")
    #  ~jname = "logBayes_V1.json"
    #  ~jname = "/home/home4/institut_1b/nattland/DNN_ttbar/BayesOptLogs/2018/Run3/logBayes_V1.json"
    jname = "/home/home4/institut_1b/nattland/DNN_ttbar/BayesOptLogs/2018/Run1/logBayes_V1.json"
    #  ~jname = "HParaLogs/logBayes_C3_4.json"
    #  ~jname = "HParaLogs/logBayes_C3_4.json"
    #  ~table = readTable(jname)
    #  ~table = readTable("HParaLogs/logBayes_comb_1_4_C1_3.json")
    table = readTable(jname)
    #  ~table = readTableNoLamb(jname)
    #  ~prntTable(lossTable)
    #  ~maxv=-1
    #  ~tab1 = readTable("HParaLogs/logBayes_C3_3.json")
    #  ~tab2 = readTable("HParaLogs/logBayes_C5_4.json")
    #  ~tab3 = readTable("HParaLogs/logBayes_C2_4.json")
    #  ~prntTable(table)
    
    lossDF = pd.DataFrame({"val_loss": -1*table[:,0], "alpha": table[:,1], "batch_size": table[:,2], "dropout": table[:,3], "lambda": table[:,4], "learn_rate": table[:,5], "n_layers": table[:,6], "n_nodes": table[:,7]})
    #  ~lossDF = pd.DataFrame({"val_loss": -1*table[:,0], "alpha": table[:,1], "batch_size": table[:,2], "dropout": table[:,3], "learn_rate": table[:,4], "n_layers": table[:,5], "n_nodes": table[:,6]})
    
    #  ~plotDF(lossDF)
    #  ~plotDFv2(lossDF)
    plotDFv2(lossDF)
    #  ~plotLoss([tab1[:,0], tab2[:,0]], jname)
