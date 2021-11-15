import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import json
#  ~from BayesianOptimization.bayes_opt import BayesianOptimization
#  ~from BayesianOptimization.bayes_opt.logger import JSONLogger
#  ~from BayesianOptimization.bayes_opt.event import Events
#  ~from BayesianOptimization.bayes_opt.util import load_logs


def plotLoss(pathList):
    losses, steps = [], []
    for path in pathList:
        templosses, tempsteps = [], []
        with open(path) as f:
            for line in f.readlines()[1:]:
                tempsteps.append(line.split(",")[1])
                templosses.append(line.split(",")[2][:-1])
        losses.append(templosses)
        steps.append(tempsteps)
    steps = np.array(steps, dtype=int)
    losses = np.array(losses, dtype=float)
    print(ssd.jensenshannon(losses[0], losses[1]))
    #  ~js_dist = ssd.jensenshannon(losses[0], losses[1])
    #  ~js_dist2 = ssd.jensenshannon(0.3*losses[0], losses[1])
    #  ~print(js_dist,js_dist2)
    plt.figure("Loss Comparison Zoom")
    plt.plot(steps[0], losses[0], color="red", ls="-", label="dropout: 0.3, $\lambda$: 0 (training)")
    plt.plot(steps[1], losses[1], color="blue", ls="-", label="dropout: 0.4, $\lambda$: 0 (validation)")
    #  ~plt.xlim(10,50)
    plt.ylim(19,29)
    plt.grid()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    #  ~plt.tight_layout()
    #  ~plt.savefig("LossPlots/Loss_Comp_Regu.pdf")

def fit_test(x,y):
    #  ~x=y
    return -x ** 2 - (y - 1) ** 2 + 1 + 4*(np.random.random()-0.5)

def hyperOpt():
    pbounds = {'x': (2, 4), 'y': (-3, 3)}

    optimizer = BayesianOptimization(
        f=fit_test,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=30,
    )
    

    logger = JSONLogger(path="HyperParameterResuls/HParaLogs/logWtest.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)


    #  ~load_logs(optimizer, 
    load_logs(optimizer, logs=["HyperParameterResuls/HParaLogs/logWrite.json"]);
    optimizer.maximize(init_points=3,n_iter=6)



    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t target: {}, x: {}, y: {}".format(i, res["target"], res["params"]["x"], res["params"]["y"]))

    print(optimizer.max)



def GridSearch():
    lrVals = np.linspace(-3,3,10)
    gridName="Testing"
    results=[]
    results.append(["a","b"])
    #  ~results.append(["ValLoss", "MSE"])
    for lr in lrVals:
        results.append(fit_test(3,lr))
    results=np.array(results)
    print(results)
    pd.DataFrame(results).to_csv("HyperParameterResuls/"+gridName+".csv", header=None)


def readSearch():
    #  ~vals = np.array([0.00001,0.00002,0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05])    #lr
    #  ~vals = np.array([0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8])   #dout
    vals = [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]  #lamb
    #  ~vals = [250, 500, 750, 1000, 2000, 5000, 7500, 10000, 25000, 50000] #batch
    #  ~vals = [1, 2, 3, 4, 5, 6, 7, 8]   #nLayers
    #  ~vals = [1./8, 1./4, 1./2, 1., 2., 4.]       #nodeFacs
    #  ~vals = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]   #alph
    resArray=[]
    resArray2=[]
    resArray1=[]
    #  ~fileName="HyperParameterResuls/lrVals/lrVals_opt1.csv"
    #  ~fileName1="HyperParameterResuls/lrVals/lrValsV2.csv"
    #  ~fileName="HyperParameterResuls/doutVals/doutVals_opt1.csv"
    #  ~fileName1="HyperParameterResuls/doutVals/doutValsV2.csv"
    fileName="HyperParameterResuls/lambVals/lambVals_opt1.csv"
    #  ~fileName = "HyperParameterResuls/batchVals/batchVals_opt1.csv"
    #  ~fileName = "HyperParameterResuls/nLayVals/nLayVals_opt1.csv"
    #  ~fileName = "HyperParameterResuls/nodeVals/nodeVals_opt1.csv"
    #  ~fileName = "HyperParameterResuls/alphVals/alphVals_opt1.csv"
    with open(fileName) as f:
        for line in f.read().split("\n"):
            resArray.append(line.split(","))
    #  ~with open(fileName1) as f:
        #  ~for line in f.read().split("\n"):
            #  ~resArray2.append(line.split(","))
    with open(fileName[:-9]+".csv") as f:
        for line in f.read().split("\n"):
            resArray2.append(line.split(","))
    labels=["logcosh_val","met_MSE_val","metVec_mean_val","jensenShannon_met_val","chisquare_met_val", "purity last bin", "err purity last bin", "purity prelast bin", "err purity prelast bin", "stability last bin", "err stability last bin", "stability prelast bin", "err stability prelast bin"]
    colors=["b","r","g","orange","purple", "navy", "navy", "cornflowerblue", "cornflowerblue", "darkgreen", "darkgreen", "lawngreen", "lawngreen"]
    values=np.array(resArray[:-1], dtype=float)
    values1=np.array(resArray1[:-1], dtype=float)
    #  ~print(np.shape(values), values[:,12])
    values2=np.array(resArray2[:-1], dtype=float)
    #  ~title="learning rate=0.0001, droupout=0.35, "+r"$\lambda=0.05$"+", 6 layers"
    xlab=r"$\lambda$"
    #  ~title="learnrate=0.001, "+r"$\lambda=0.05$"
    #  ~print(values)
    logb = True
    for i in range(5):
        plt.figure()
        #  ~plt.suptitle(title, fontsize=12, ha="left", x=0.12, y=0.99)
        plt.plot(vals, values[:,i], label=labels[i], color=colors[i])
        if logb: plt.xscale("log")
        plt.grid()
        plt.legend()
        plt.ylabel("target")
        plt.xlabel(xlab)
        plt.tight_layout(pad=0.2)
        plt.savefig(fileName[:-4]+"_"+labels[i][:-4]+".pdf")    
        
    plt.figure()
    #  ~plt.suptitle(title, fontsize=12, ha="left", x=0.12, y=0.99)
    for i in range(5):
        plt.plot(vals, (values[:,i]-np.min(values[:,i]))/sum(values[:,i]-np.min(values[:,i])), label=labels[i], color=colors[i])
    if logb: plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.ylabel(r"$\frac{(f-min(f))}{sum(f-min(f))}$")
    plt.xlabel(xlab)
    plt.tight_layout(pad=0.2)
    plt.savefig(fileName[:-4]+"_Comp.pdf")
    
    plt.figure()
    #  ~plt.suptitle(title, fontsize=12, ha="left", x=0.12, y=0.99)
    for i in range(3):
        plt.plot(vals, (values[:,i])/sum(values[:,i]), label=labels[i], color=colors[i])
    plt.plot(vals, (values2[:,0])/sum(values2[:,0]), label=labels[0]+" before", color=colors[0], linestyle="dashed")
    if logb: plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.ylabel("normalized target")
    plt.xlabel(xlab)
    plt.tight_layout(pad=0.2)
    plt.savefig(fileName[:-4]+"_CompLoss.pdf")
    
    plt.figure()
    #  ~plt.suptitle(title, fontsize=12, ha="left", x=0.12, y=0.99)
    for i in range(3,5):
        plt.plot(vals, (values[:,i])/sum(values[:,i]), label=labels[i], color=colors[i])
    if logb: plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.ylabel("normalized target")
    plt.xlabel(xlab)
    plt.tight_layout(pad=0.2)
    plt.savefig(fileName[:-4]+"_CompMet.pdf")
    #  ~plt.savefig(fileName[:-4]+"_"+labels[i][:-4]+".pdf")
    #  ~plt.show()
    #  ~print(labels[i][:-4])
    plt.figure()
    for i in [5,7,9,11]:
        #  ~plt.plot(vals, (values[:,i]), label=labels[i], color=colors[i])
        plt.errorbar(vals, (values[:,i]), yerr=values[:,i+1], label=labels[i], color=colors[i])
    if logb: plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.ylabel("target")
    plt.xlabel(xlab)
    plt.tight_layout(pad=0.2)
    plt.savefig(fileName[:-4]+"_PurStab.pdf")
    
def ConvTojson():
    lrvals = [0.00001,0.00002,0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05]   #lr
    #  ~doutvals = [0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]   #dout
    #  ~lambvals = [0.,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]  #lamb
    #  ~batchvals = [250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000] #batch
    #  ~nLayvals = [1, 2, 3, 4, 5, 6, 7, 8]   #nLayers
    #  ~nodevals = [1./8, 1./4, 1./2, 1., 2., 4.]       #nodeFacs
    #  ~alphvals = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]   #alph
    resArray=[]
    #  ~resArray2=[]
    #  ~fileName="HyperParameterResuls/lrVals/lrVals.csv"
    fileName="HyperParameterResuls/lrVals/lrValsV2.csv"
    #  ~fileName2="HyperParameterResuls/doutVals/doutVals.csv"
    #  ~fileName="HyperParameterResuls/lambVals/lambVals.csv"
    #  ~fileName="HyperParameterResuls/doutVals/doutValsV2.csv"
    #  ~fileName = "HyperParameterResuls/batchVals/batchVals.csv"
    #  ~fileName = "HyperParameterResuls/nLayVals/nLayVals.csv"
    #  ~fileName = "HyperParameterResuls/nLayVals/nLayVals_opt1.csv"
    with open(fileName) as f:
        for line in f.read().split("\n"):
            resArray.append(line.split(","))
    values=np.array(resArray[:-1], dtype=float)
    stdvals1 = [0.0001, 0.35, 0.05, 5000, 6, 1, 0.2]
    for i,val in enumerate(values):
        data=dict()
        data["target"]=val
        data["params"]={
            "lr": x[i], 
            "dout": y[i],
            "lamb": y[i],
            "batch": y[i],
            "nLayers": y[i],
            "nodeFac": y[i],
            "alpha": y[i],
        }
        data["datetime"] = {
            "datetime": 0.0,
            "elapsed": 0.0,
            "delta": 0.0,
        }
        print(data)
        with open("HyperParameterResuls/HParaLogs/logGrid.json", "a") as f:
            f.write(json.dumps(data)+"\n")
    
def readnPrint():
    resArray=[]
    with open("HyperParameterResuls/lrVals/lrV2vals") as f:
        for line in f.read().split("]\n ["):
            resArray.append(line.split())
    resArray=np.array(resArray)
    resArray[0,0]=resArray[0,0][2:]
    resArray[-1,-1]=resArray[-1,-1][:-2]
    resArray=np.array(resArray, dtype=float)
    np.savetxt("HyperParameterResuls/lrVals/lrValsV2.csv", resArray, delimiter=",")

#############################################################

if __name__ == "__main__":
    pathList = ["LossPlots/run-Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1421genMETweighted_validation-tag-epoch_loss.csv",
    "LossPlots/run-Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211025-1130genMETweighted_validation-tag-epoch_loss.csv"]
    #  ~plotLoss(pathList)
    #  ~GridSearch()
    readSearch()
    #  ~hyperOpt()
    #  ~readnPrint()
    #  ~ConvTojson()
