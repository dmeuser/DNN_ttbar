import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import json
#  ~import home.home4.institut_1b.nattland.CMSSW_10_5_0.src.CombineHarvester.CombineTools.plotting as plotComb
#  ~import CombineHarvester.CombineTools.plotting as plotComb
#  ~from BayesianOptimization.bayes_opt import BayesianOptimization
#  ~from BayesianOptimization.bayes_opt.logger import JSONLogger
#  ~from BayesianOptimization.bayes_opt.event import Events
#  ~from BayesianOptimization.bayes_opt.util import load_logs


def plotLoss(pathList,plotName):
    losses, steps = [], []
    for path in pathList:
        templosses, tempsteps = [], []
        with open(path) as f:
            for line in f.readlines()[1:]:
                tempsteps.append(line.split(",")[1])
                templosses.append(line.split(",")[2][:-1])
        losses.append(templosses)
        steps.append(tempsteps)
    stepList = [np.array(step, dtype=int) for step in steps]
    lossList = [np.array(loss, dtype=float) for loss in losses]
    #  ~print(stepList, lossList)
    #  ~print(ssd.jensenshannon(losses[0], losses[1]))
    #  ~js_dist = ssd.jensenshannon(losses[0], losses[1])
    #  ~js_dist2 = ssd.jensenshannon(0.3*losses[0], losses[1])
    #  ~print(js_dist,js_dist2)
    #  ~fig, ax  = plt.subplots()
    plt.figure("testint")
    #  ~ax.plot(stepList[0], lossList[0], color="blue", ls="-", label="oversampling with minimal noise (training)")
    #  ~ax.plot(stepList[1], lossList[1], color="blue", ls="--", label="oversampling with minimal noise (validation)")
    #  ~plt.plot(stepList[0], lossList[0], color="blue", ls="-", label="undersampling (training)")
    #  ~plt.plot(stepList[1], lossList[1], color="blue", ls="--", label="undersampling (validation)")
    #  ~plt.plot(stepList[2], lossList[2], color="red", ls="-", label="over- and undersampling (training)")
    #  ~plt.plot(stepList[3], lossList[3], color="red", ls="--", label="over- and undersampling (validation)")
    #  ~plt.plot(stepList[4], lossList[4], color="black", ls="-", label="genMET reweighting (training)")
    #  ~plt.plot(stepList[5], lossList[5], color="black", ls="--", label="genMET reweighting (validation)")
    #  ~plt.xlim(0,73)
    #  ~plt.ylim(15,26)
    #  ~ax.grid()
    #  ~ax.set_ylabel("Loss")
    #  ~ax.set_xlabel("Epoch")
    #  ~ax.legend()
    #  ~ax = plt.gca()
    #  ~pos1 = ax.get_position()
    #  ~print(pos1)
    #  ~pos1, "testint"
    #  ~plt.text(0.5, 0.2, r"testing123")
    #  ~plt.show()
    #  ~plt.tight_layout()
    #  ~fig.savefig(plotName)

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
    labels=["loss_val","met_MSE_val","metVec_mean_val","jensenShannon_met_val","chisquare_met_val", "purity last bin", "err purity last bin", "purity prelast bin", "err purity prelast bin", "stability last bin", "err stability last bin", "stability prelast bin", "err stability prelast bin"]
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
    
def readnPrint(filename):
    resArray=[]
    with open(filename) as f:
        for line in f.read().split("\n"):
            resArray.append(line.split(","))
    resArray=np.array(resArray[:-1], dtype=float)[1::2]
    #  ~print(resArray
    lowArr = np.where(resArray[:,1]<20, True, False)
    valStrings=["loss", "logcosh", "MSE", "MSE vectorial", "Jensonshannon", "chisqr"]
    for col in range(6):
        #  ~arr=resArray[:,col]
        arr=resArray[:,col][lowArr][2:]
        #  ~print(arr)
        #  ~print(arr)
        print(valStrings[col]+"\nmean: {:10.5g} +- {:<10.5g} ({:.2g} %)".format(np.mean(arr),np.std(arr), 100*np.std(arr)/np.mean(arr)))
        #  ~print(r"{:.6g} & {:.2g} & {:.2g}$\%$\\".format(np.mean(arr),np.std(arr),np.std(arr)*100/np.mean(arr)))
        #  ~print(arr[lowArr])

def readnPrint2(filename):
    resArray=[]
    with open(filename) as f:
        for line in f.read().split("\n"):
            resArray.append(line.split(","))
    resArray= (np.array(resArray, dtype=float))
    for col in range(2):
        arr=resArray[:,col]
        print("mean: {:10.5g} +- {:<10.5g} ({:.2g} %)".format(np.mean(arr),np.std(arr), 100*np.std(arr)/np.mean(arr)))

def readnPrint3(filename):
    resArray=[]
    with open(filename) as f:
        for line in f.readlines():
            tempArr = []
            tempArr.append(line.split(",")[0][11:])
            tempArr.append(line.split(",")[1][20:])
            tempArr.append(line.split(",")[2][10:])
            tempArr.append(line.split(",")[3][9:])
            tempArr.append(line.split(",")[4][7:])
            tempArr.append(line.split(",")[5][11:])
            tempArr.append(line.split(",")[6][12:-1])
            resArray.append(tempArr)

    resArray= (np.array(resArray, dtype=float))
    resArray = resArray[resArray[:, 0].argsort()]
    #  ~minVal = np.argmax(resArray[:,0])
    print(resArray[-3:,:])
    
#############################################################

if __name__ == "__main__":
    path = "/home/home4/institut_1b/nattland/cerncluster/DNN_ttbar/BayesOptLogs/2018/Run3/logBayes_V1.json"
    readnPrint3(path)
