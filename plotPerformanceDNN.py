from ROOT import TMVA, TFile, TTree, TCut, TMath, TH1F
from subprocess import call
import argparse
from os.path import isfile
import sys
import struct
import datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#  ~os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model

import scipy.spatial.distance as ssd
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
import shap
import seaborn as sns
import tensorflow as tf
from DNN_funcs import getInputArray_allBins_nomDistr, getMETarrays



# ----------------------------------------------------------------------
# Following functions meanFunc... and rmsFunc... are for evaluation of rms and mean of difference between genMET and recoMET while taking SFweights into account; Input "x" is a 2 dimensional array, whith one column being the difference, one column being the SF weights
def meanFunc2d(x):
    y = np.array(list(map(np.array, np.array(x))))
    res = np.average(y[:,0], weights=y[:,1])
    return res

def rmsFunc2d(x):
    av = meanFunc2d(x)
    y = np.array(list(map(np.array, np.array(x))))
    res = np.sqrt(np.average((y[:,0]-av)**2, weights=y[:,1]))
    return res

def meanErr2d(x):
    y = np.array(list(map(np.array, np.array(x))))
    return 2*rmsFunc2d(x)/np.sqrt(y.shape[0])

def rmsFUNC(x):
    return np.sqrt(np.mean(np.square(x-np.mean(x))))

def meanErr(x):
    return 2*np.std(x)/np.sqrt(len(x))
# ----------------------------------------------------------------------



def get_inputs(year,dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):
    
    if not os.path.exists("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]):    # create output folder for plots if not available
        os.makedirs("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1])
    
    # get inputs
    if genMETweighted:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals, train_weights, val_weights, test_weights = getInputArray_allBins_nomDistr(year,dataPath,inputVars,targetName,target,treeName,update=updateInput,normalize=normalize,standardize=False,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
        
    else:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals = getInputArray_allBins_nomDistr(year,dataPath,inputVars,targetName,target,treeName,update=updateInput,normalize=normalize,standardize=False,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
    
    model = load_model(modelPath+".h5")
    
    print(train_y.keys())
    
    if standardize:
        train_x_scaler = StandardScaler()
        train_x_std = pd.DataFrame(train_x_scaler.fit_transform(train_x.values), index=train_x.index, columns=train_x.columns)
        
        train_y_scaler = StandardScaler()
        train_y_std = pd.DataFrame(train_y_scaler.fit_transform(train_y.values), index=train_y.index, columns=train_y.columns)
        
        val_x_scaler = StandardScaler()
        val_x_std = pd.DataFrame(val_x_scaler.fit_transform(val_x.values), index=val_x.index, columns=val_x.columns)
        
        val_y_scaler = StandardScaler()
        val_y_std = pd.DataFrame(val_y_scaler.fit_transform(val_y.values), index=val_y.index, columns=val_y.columns)
        
        y_hat_train = model.predict(train_x_std,use_multiprocessing=True)
        y_hat_val = model.predict(val_x_std,use_multiprocessing=True)
        y_hat_train = train_y_scaler.inverse_transform(y_hat_train)
        y_hat_val = val_y_scaler.inverse_transform(y_hat_val)
    else:
        y_hat_train = model.predict(train_x,use_multiprocessing=True)
        y_hat_val = model.predict(val_x,use_multiprocessing=True)
        y_hat_test = model.predict(test_x,use_multiprocessing=True)

    
    train_x["DNN_1"]=[row[0] for row in y_hat_train]
    train_x["DNN_2"]=[row[1] for row in y_hat_train]
    train_x[target[0]]=train_y[target[0]]
    train_x[target[1]]=train_y[target[1]]
    train_x["PuppiMET_xy_phi"]=train_metVals["PuppiMET_xy_phi"]
    train_x["genMET_phi"]=train_metVals["genMET_phi"]
    train_x["SF"]=train_metVals["SF"]
    
    #  ~train_x.to_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/trainResults.pkl")
    
    val_x["DNN_1"]=[row[0] for row in y_hat_val]
    val_x["DNN_2"]=[row[1] for row in y_hat_val]
    val_x[target[0]]=val_y[target[0]]
    val_x[target[1]]=val_y[target[1]]
    val_x["PuppiMET_xy_phi"]=val_metVals["PuppiMET_xy_phi"]
    val_x["genMET_phi"]=val_metVals["genMET_phi"]
    val_x["SF"]=val_metVals["SF"]
    
    #  ~val_x.to_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/valResults.pkl")
    
    test_x["DNN_1"]=[row[0] for row in y_hat_test]
    test_x["DNN_2"]=[row[1] for row in y_hat_test]
    test_x[target[0]]=test_y[target[0]]
    test_x[target[1]]=test_y[target[1]]
    test_x["PuppiMET_xy_phi"]=test_metVals["PuppiMET_xy_phi"]
    test_x["genMET_phi"]=test_metVals["genMET_phi"]
    test_x["SF"]=test_metVals["SF"]
    
    #  ~test_x.to_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/testResults.pkl")
    return train_x, val_x, test_x


def print_targets(year,dataPath,inputVars,modelPath,treeName,targetName,target,title,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):

    train_x, val_x, test_x = get_inputs(year,dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=updateInput,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
    
    binN=100
    sampleNames=["training sample", "validation sample", "test sample"]
    
    print("\n\nDNN NR. ",modelNr)
    print(title)
    
    for i_sample,data_x in enumerate([train_x, val_x, test_x]):
        #  ~data_x["DNN_MET_X"]=data_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]-data_x["DNN_1"]
        #  ~data_x["DNN_MET_Y"]=data_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]-data_x["DNN_2"]
        #  ~data_x["DNN_MET"]=np.sqrt(data_x["DNN_MET_X"]**2+data_x["DNN_MET_Y"]**2)
        #  ~data_x["genMET_X"]=data_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]-data_x[target[0]]
        #  ~data_x["genMET_Y"]=data_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]-data_x[target[1]]
        #  ~data_x["genMET"]=np.sqrt(data_x["genMET_X"]**2+data_x["genMET_Y"]**2)
        
        y_pred = [data_x["DNN_1"], data_x["DNN_2"]]
        y_true = [data_x[target[0]], data_x[target[1]]]
        l = tf.keras.losses.LogCosh()
        logcosh = l(y_true, y_pred).numpy()
        
        #  ~met_MSE=np.mean((data_x["DNN_MET"]-data_x["genMET"])**2)
        #  ~metVec_mean=np.mean((data_x["DNN_MET_X"]-data_x["genMET_X"])**2+(data_x["DNN_MET_Y"]-data_x["genMET_Y"])**2)
        
        #  ~min_x, max_x = 0., 400.
        #  ~countsDNN,bins = np.histogram(np.clip(data_x["DNN_MET"],min_x+0.01,max_x-0.01),bins=binN,range=(min_x,max_x))
        #  ~countsGen,bins = np.histogram(np.clip(data_x["genMET"],min_x+0.01,max_x-0.01),bins=binN,range=(min_x,max_x))
        #  ~countsDNN, countsGen = np.array(countsDNN), np.array(countsGen)
        

        #  ~print("\n\n"+sampleNames[i_sample]+"\n")
        #  ~jensenShannon_met = ssd.jensenshannon(countsDNN, countsGen)
        #  ~chisquare_met = sum((countsDNN-countsGen)**2/countsGen)/binN
        
        print("loss (logcosh w/o regularisation term): {0:.5g}".format(logcosh))
        #  ~print("MSE met difference absolute: {0:.5g}".format(met_MSE))
        #  ~print("mean met difference vectorial: {0:.5g}".format(metVec_mean))
        #  ~print("Jensen-Shannon distance met: {0:.5g}".format(jensenShannon_met))
        #  ~print("chi square met distribution: {0:.5g}".format(chisquare_met))


def plot_Output(year,dataPath,inputVars,modelPath,treeName,targetName,target,title,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):
    
    train_x, val_x, test_x = get_inputs(year,dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=updateInput,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
    
    sampleNames=["train", "val", "test"]
    
    for i_sample,data_x in enumerate([train_x, val_x, test_x]):
        SF_weights = data_x["SF"]
        
        # Compare DNN and target
        # First three plots not yet working with SF_weights
        #  ~plt.figure()
        #  ~plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
        #  ~data_x[["PuppiMET_xy*cos(PuppiMET_xy_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)"]].plot.hist(alpha=0.5,bins=500,density=True,weights=[SF_weights, SF_weights])
        #  ~plt.tight_layout(pad=0.1)
        #  ~plt.savefig("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/"+sampleNames[i_sample]+"_Puppi_xy_"+str(modelNr)+".pdf")
        
        #  ~plt.figure()
        #  ~plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
        #  ~data_x[["DNN_1",target[0]]].plot.hist(alpha=0.5,bins=500,density=True,weights=[SF_weights, SF_weights])
        #  ~plt.tight_layout(pad=0.1)
        #  ~plt.savefig("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/"+sampleNames[i_sample]+"_x_"+str(modelNr)+".pdf")
        
        #  ~plt.figure()
        #  ~plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
        #  ~data_x[["DNN_2",target[1]]].plot.hist(alpha=0.5,bins=500,density=True,weights=[SF_weights, SF_weights])
        #  ~plt.tight_layout(pad=0.1)
        #  ~plt.savefig("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/"+sampleNames[i_sample]+"_y_"+str(modelNr)+".pdf")

        
        # Compare corrected MET to genMET X 
        data_x["genMET_X"]=data_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]-data_x[target[0]]
        data_x["DNN_MET_X"]=data_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]-data_x["DNN_1"]
        
        min=data_x["genMET_X"].min()
        max=data_x["genMET_X"].max()
        plt.figure()
        plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
        data_x["genMET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_X",weights=SF_weights)
        data_x["DNN_MET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_X",weights=SF_weights)
        plt.legend()
        plt.tight_layout(pad=0.1)
        plt.savefig("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/METdistr_x_"+sampleNames[i_sample]+"_"+str(modelNr)+".pdf")
        
        # Compare corrected MET to genMET Y
        data_x["genMET_Y"]=data_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]-data_x[target[1]]
        data_x["DNN_MET_Y"]=data_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]-data_x["DNN_2"]
        
        min=data_x["genMET_Y"].min()
        max=data_x["genMET_Y"].max()
        plt.figure()
        data_x["genMET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_Y",weights=SF_weights)
        data_x["DNN_MET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_Y",weights=SF_weights)
        plt.legend()
        plt.tight_layout(pad=0.1)
        plt.savefig("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/METdistr_y_"+sampleNames[i_sample]+"_"+str(modelNr)+".pdf")
        
        # Compare corrected MET to genMET pT    
        min_x=0
        max_x=400
        subRatio = dict(height_ratios=[5,3])
        binN = 100
        
        data_x["genMET"]=np.sqrt(data_x["genMET_X"]**2+data_x["genMET_Y"]**2)
        data_x["DNN_MET"]=np.sqrt(data_x["DNN_MET_X"]**2+data_x["DNN_MET_Y"]**2)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=subRatio)
        countsDNN,bins = np.histogram(data_x["DNN_MET"],bins=binN,range=(min_x,max_x))
        countsGen,bins = np.histogram(data_x["genMET"],bins=binN,range=(min_x,max_x))
        ax1.hist(data_x["DNN_MET"],alpha=0.5,bins=binN,range=(min_x,max_x),density=True,label="DNN", color="b",weights=SF_weights)
        ax1.hist(data_x["genMET"],alpha=0.5,bins=binN,range=(min_x,max_x),density=True,label="gen", color="r",weights=SF_weights)
        countsDNN, countsGen = np.array(countsDNN), np.array(countsGen)
        ratioErrs = countsDNN/countsGen*np.sqrt(countsDNN/np.square(countsDNN)+countsGen/np.square(countsGen))
        ax2.errorbar(0.5*(bins[1:]+bins[:-1]), countsDNN/countsGen, yerr=ratioErrs, color="b", fmt=".", capsize=2, elinewidth=1, markersize=1.5, ls="none")
        ax2.set_xlabel("MET [GeV]")
        ax2.axhline(1, color="r", linewidth=1, alpha=0.5)
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax1.set_ylabel("Normalized Counts")
        ax2.set_ylabel("ratio")
        ax1.set_ylim(0,0.013)
        ax2.set_ylim(0.49,2.01)
        ax1.legend()
        ax1.grid()
        makeCMStitle(year)
        ax2.grid()
        plt.tight_layout(pad=0.1)
        plt.savefig("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/METdistr_pT_"+sampleNames[i_sample]+"_"+str(modelNr)+".pdf")
        
        # Compare resolution for corrected MET and Puppi for whole genMET range ("tot"), genMET<=200GeV and genMET<200GeV
        split_str = ["tot",">=200 GeV", "<200 GeV"]
        # Calculate phi angle between DNN and gen
        dPhi_DNN_gen_arr = np.array([(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_x["genMET_phi"]), (2*np.pi+np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_x["genMET_phi"])), (-2*np.pi+np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_x["genMET_phi"]))])
        data_x["dPhi_DNN_gen"] = dPhi_DNN_gen_arr.flatten()[np.arange(np.shape(dPhi_DNN_gen_arr)[1]) + np.abs(dPhi_DNN_gen_arr).argmin(axis=0)*dPhi_DNN_gen_arr.shape[1]]
        # Calculate phi angle between Puppi and gen
        dPhi_Puppi_gen_arr = np.array([(data_x["PuppiMET_xy_phi"]-data_x["genMET_phi"]), (2*np.pi+(data_x["PuppiMET_xy_phi"]-data_x["genMET_phi"])), (-2*np.pi+(data_x["PuppiMET_xy_phi"]-data_x["genMET_phi"]))])
        data_x["dPhi_Puppi_gen"] = dPhi_Puppi_gen_arr.flatten()[np.arange(np.shape(dPhi_Puppi_gen_arr)[1]) + np.abs(dPhi_Puppi_gen_arr).argmin(axis=0)*dPhi_Puppi_gen_arr.shape[1]]
        
        for i,train_temp in enumerate([data_x, data_x[data_x["genMET"]>=200], data_x[data_x["genMET"]<200]]):
            train_temp["PuppiMET_xy"]=np.sqrt(train_temp["PuppiMET_xy*cos(PuppiMET_xy_phi)"]**2+train_temp["PuppiMET_xy*sin(PuppiMET_xy_phi)"]**2)
            train_temp["genMET-PuppiMET_xy"]=train_temp["genMET"]-train_temp["PuppiMET_xy"]
            train_temp["genMET-DNN_MET"]=train_temp["genMET"]-train_temp["DNN_MET"]
            # Calculate mean and std while taking SF_weights into account
            # MET difference
            tempMeanPuppi = np.average(train_temp["genMET-PuppiMET_xy"], weights=train_temp["SF"])
            tempStdPuppi = np.sqrt(np.average((train_temp["genMET-PuppiMET_xy"]-tempMeanPuppi)**2, weights=train_temp["SF"]))
            tempMeanDNN = np.average(train_temp["genMET-DNN_MET"], weights=train_temp["SF"])
            tempStdDNN = np.sqrt(np.average((train_temp["genMET-DNN_MET"]-tempMeanDNN)**2, weights=train_temp["SF"]))
            plotStats = "Mean Puppi: {:.2f}\nStd Puppi: {:.2f}\nMean DNN: {:.2f}\nStd DNN: {:.2f}".format(tempMeanPuppi, tempStdPuppi, tempMeanDNN, tempStdDNN)
            min_x=-150
            max_x=150
            
            plt.figure()
            train_temp["genMET-DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=False,label="DNN", color="b",weights=train_temp["SF"])
            train_temp["genMET-PuppiMET_xy"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=False,label="Puppi", color="r",weights=train_temp["SF"])
            plt.text(-148, 0.8*plt.gca().get_ylim()[1], plotStats, bbox={"facecolor":"none", "pad":5})
            plt.axvline(0, color="black", linewidth=1)
            plt.xlabel("genMET-recoMET [GeV]")
            plt.ylabel("Events")
            plt.legend()
            makeCMStitle(year)
            plt.savefig("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/Resolution_pT_"+sampleNames[i_sample]+split_str[i]+"_"+str(modelNr)+".pdf")
            plt.clf()
            
            # difference in phi of MET
            min_x = np.min(train_temp["dPhi_Puppi_gen"])
            max_x = np.max(train_temp["dPhi_Puppi_gen"])
            tempMeanPuppi = np.average(train_temp["dPhi_Puppi_gen"], weights=train_temp["SF"])
            tempStdPuppi = np.sqrt(np.average((train_temp["dPhi_Puppi_gen"]-tempMeanPuppi)**2, weights=train_temp["SF"]))
            tempMeanDNN = np.average(train_temp["dPhi_DNN_gen"], weights=train_temp["SF"])
            tempStdDNN = np.sqrt(np.average((train_temp["dPhi_DNN_gen"]-tempMeanDNN)**2, weights=train_temp["SF"]))
            plotStats = "Mean Puppi: {:.2f}\nStd Puppi: {:.2f}\nMean DNN: {:.2f}\nStd DNN: {:.2f}".format(tempMeanPuppi, tempStdPuppi, tempMeanDNN, tempStdDNN)
            
            plt.figure()
            train_temp["dPhi_DNN_gen"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=False,label="DNN", color="b",weights=train_temp["SF"])
            train_temp["dPhi_Puppi_gen"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=False,label="Puppi", color="r",weights=train_temp["SF"])
            plt.text(-3, 0.8*plt.gca().get_ylim()[1], plotStats, bbox={"facecolor":"none", "pad":5})
            plt.axvline(0, color="black", linewidth=1)
            plt.xlabel(r"$\Delta\Phi(genMET,\,recoMET)$")
            plt.ylabel("Events")
            plt.legend()
            makeCMStitle(year)
            plt.savefig("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/Resolution_dPhi_"+sampleNames[i_sample]+split_str[i]+"_"+str(modelNr)+".pdf")

        
        # Compare mean diff as a function of genMET, SFweigths only implemented correctly in "genMETweighted"
        if genMETweighted:
            print("weighted plots")
            binsMET = np.linspace(0,400,21)
            data_x["bin"] = np.digitize(data_x["genMET"],bins=binsMET)
            data_x["both_DNN"] = [i for i in zip(data_x["genMET-DNN_MET"], data_x["SF"])]
            data_x["both_Puppi"] = [i for i in zip(data_x["genMET-PuppiMET_xy"], data_x["SF"])]
            res_DNN_MET = data_x.groupby("bin")["both_DNN"].agg([meanFunc2d,rmsFunc2d,meanErr2d])
            res_PuppiMET_xy = data_x.groupby("bin")["both_Puppi"].agg([meanFunc2d,rmsFunc2d,meanErr2d])
            res_DNN_MET["metBins"] = binsMET
            plt.figure()
            plt.errorbar(binsMET, res_DNN_MET["meanFunc2d"], yerr=res_DNN_MET["meanErr2d"], color="b", label=None, ls="none", capsize=3)
            plt.step(binsMET, res_DNN_MET["meanFunc2d"], color="b", linewidth=1, where="mid", label="DNN mean")
            plt.step(binsMET, res_DNN_MET["rmsFunc2d"], color="b", linewidth=1, where="mid", linestyle="--", label="DNN rms")
            plt.errorbar(binsMET, res_PuppiMET_xy["meanFunc2d"], yerr=res_PuppiMET_xy["meanErr2d"], color="r", label=None, ls="none", capsize=3)
            plt.step(binsMET, res_PuppiMET_xy["meanFunc2d"], color="r", linewidth=1, where="mid", label="Puppi mean")
            plt.step(binsMET, res_PuppiMET_xy["rmsFunc2d"], color="r", linewidth=1, where="mid", linestyle="--", label="Puppi rms")
            plt.ylabel("genMET-recoMET (GeV)")
            plt.xlabel("genMET (GeV)")
            plt.grid()
            plt.legend()
            plt.ylim(-40,80)
            makeCMStitle(year)
            plt.tight_layout(pad=0.1)
            plt.savefig("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/MeanDiff_vs_genMET_"+sampleNames[i_sample]+"_"+str(modelNr)+".pdf")
        else:
            binsMET = np.linspace(0,400,21)
            data_x["bin"] = np.digitize(data_x["genMET"],bins=binsMET)
            res_DNN_MET = data_x.groupby("bin")["genMET-DNN_MET"].agg(["mean",rmsFUNC,meanErr])
            res_PuppiMET_xy = data_x.groupby("bin")["genMET-PuppiMET_xy"].agg(["mean",rmsFUNC,meanErr])
            res_DNN_MET["metBins"] = binsMET
            plt.figure()
            plt.errorbar(binsMET, res_DNN_MET["mean"], yerr=res_DNN_MET["meanErr"], color="b", label=None, ls="none", capsize=3)
            plt.step(binsMET, res_DNN_MET["mean"], color="b", linewidth=1, where="mid", label="DNN mean")
            plt.step(binsMET, res_DNN_MET["rmsFUNC"], color="b", linewidth=1, where="mid", linestyle="--", label="DNN rms")
            plt.errorbar(binsMET, res_PuppiMET_xy["mean"], yerr=res_PuppiMET_xy["meanErr"], color="r", label=None, ls="none", capsize=3)
            plt.step(binsMET, res_PuppiMET_xy["mean"], color="r", linewidth=1, where="mid", label="Puppi mean")
            plt.step(binsMET, res_PuppiMET_xy["rmsFUNC"], color="r", linewidth=1, where="mid", linestyle="--", label="Puppi rms")
            plt.ylabel("genMET-recoMET (GeV)")
            plt.xlabel("genMET (GeV)")
            plt.grid()
            plt.legend()
            plt.ylim(-40,80)
            makeCMStitle(year)
            plt.tight_layout(pad=0.1)
            plt.savefig("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/MeanDiff_vs_genMET_"+sampleNames[i_sample]+"_"+str(modelNr)+".pdf")

        
        plt.figure()
        #  ~data_x["genMET"].plot.hist(alpha=0.3, color="blue", bins=bins, density=True, weights=train_weights, label="genMET with weights")
        #  ~data_x["genMET"].plot.hist(alpha=0.3, color="red", bins=300, density=True, weights=train_weights, label="genMET with weights")
        data_x["genMET"].plot.hist(alpha=0.6, color="b", bins=300, density=True, label="genMET "+sampleNames[i_sample]+"ing w/o weights", weights=SF_weights)
        plt.legend()
        plt.savefig("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/genMET_distr_"+sampleNames[i_sample]+"_"+str(modelNr)+".pdf")
        
        with open("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/DNNdescription", "w") as f:
            f.write(title)


def makeCMStitle(year):
    if year=="2016_preVFP":
        lumi="19.5"
    elif year=="2016_postVFP":
        lumi="16.8"
    elif year=="2017":
        lumi="41.5"
    else:
        lumi="59.8"
    ax = plt.gca()
    ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
    ax.text(0.,1.,r"           $\,$Simulation$\,\bullet\,$Private work",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', style="italic", fontsize=10, color=(0.3,0.3,0.3))
    ax.text(1.,1.,lumi+r"$\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)


def plot_Purity(year,dataPath,inputVars,modelPath,treeName,targetName,target,title,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):
    
    # To achieve sensible statistics in high MET bins, the Purity is usually evaluated for the whole sample, calling it "training" here, arbitrairily. For this, the validation- and testsize are set to 0.01 in genMETarrays()
    train_metVals, val_metVals, test_metVals = getMETarrays(year,dataPath,inputVars,modelPath,treeName,targetName,target,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)
    
    # Choose wether to used optimized binnings by Fabian ("True") or old binnings ("False")
    optBins = True
    if optBins:
        metBins = np.array([0,50,70,100,130,160,200,400])
        dphiBins = np.array([0,0.64,1.2,3.15])
    else:
        metBins = np.array([0,40,80,120,160,230,400])
        dphiBins = np.array([0,0.7,1.4,3.15])
        
    vmax = 0.8 # maximum value in the scale of the colorbar
    
    for (data_metVals, sampName) in [(train_metVals, "train")]:
    #  ~for (data_metVals, sampName) in [(train_metVals, "train"), (val_metVals, "val"), (test_metVals, "test")]:
        
        # using last bin as overflow bin
        
        met_gen = np.clip(data_metVals["PtNuNu"], metBins[0], metBins[-1])
        dphi_gen = data_metVals["dPhi_PtNuNuNearLep"]
        
        met_reco_DNN = np.clip(data_metVals["DNN_MET"], metBins[0], metBins[-1])
        dphi_reco_DNN = data_metVals["DNN_dPhiMetNearLep"]

        met_reco_Puppi = np.clip(data_metVals["PuppiMET_xy"], metBins[0], metBins[-1])
        dphi_reco_Puppi = data_metVals["dPhi_PuppiNearLep"]
        
        
        for (met_reco, dphi_reco, dnnName) in [(met_reco_DNN, dphi_reco_DNN, "DNN",), (met_reco_Puppi, dphi_reco_Puppi, "Puppi")]:
            
            # creating histograms according to generated met/dphi (Gen), reconstructed met/dphi, and events, where both gen and reco are in the same bin
            
            histo2D_Gen, xedges, yedges = np.histogram2d(met_gen, dphi_gen, bins=[metBins, dphiBins])
            histo2D_Reco, xedges, yedges = np.histogram2d(met_reco, dphi_reco, bins=[metBins, dphiBins])
            histo2D_Both = np.copy(histo2D_Gen)
            
            nB = (len(metBins)-1)*(len(dphiBins)-1)
            histoResponse, edges = np.histogramdd(np.array([dphi_reco, met_reco, dphi_gen, met_gen]).T, bins=(dphiBins, metBins, dphiBins, metBins))
            histoResponse = histoResponse.reshape(len(metBins)-1,len(dphiBins)-1,-1)
            histoResponse = histoResponse.reshape(-1,nB)
            histoResponse = histoResponse / histoResponse.sum(axis=1)[:,np.newaxis]
            
            for i in range(len(metBins)-1):
                for j in range(len(dphiBins)-1):
                    temp1 = np.where((met_gen>metBins[i]) & (met_gen<=metBins[i+1]) & (dphi_gen>dphiBins[j]) & (dphi_gen<=dphiBins[j+1]), True, False)
                    temp2 = np.where((met_reco>metBins[i]) & (met_reco<=metBins[i+1]) & (dphi_reco>dphiBins[j]) & (dphi_reco<=dphiBins[j+1]), True, False)
                    histo2D_Both[i,j] = sum(np.where(temp1 & temp2, 1, 0))
                    
                    
            histo2D_Gen = histo2D_Gen.T
            histo2D_Reco = histo2D_Reco.T
            histo2D_Both = histo2D_Both.T
            
            print("{} Reco {} Last Bin Count: {}".format(dnnName, sampName, histo2D_Reco[-1,-1]))
            print("{} Both {} Last Bin Count: {}".format(dnnName, sampName, histo2D_Both[-1,-1]))
            
            # Plotting Purity
            
            Xbins, Ybins = np.meshgrid(metBins, dphiBins)
            metBinsC = (metBins[1:]+metBins[:-1])*0.5
            dphiBinsC = (dphiBins[1:]+dphiBins[:-1])*0.5
            
            fig, ax = plt.subplots(1,1)
            mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both/histo2D_Reco, vmin=0., vmax=vmax)
            for i,phiarr in enumerate(histo2D_Both/histo2D_Reco):
                for j,vali in enumerate(phiarr):
                    if optBins: ax.text(metBinsC[j]-15, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=10, color="red", fontweight="bold")
                    else: ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=14, color="red", fontweight="bold")
                    # Plotting uncertainties:
                    #  ~tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Reco[i,j])
                    #  ~ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
                    #  ~ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=7, color="red")
            cbar = fig.colorbar(mesh1, ax=ax)
            cbar.set_label("purity")
            ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
            ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
            makeCMStitle(year)
            plt.savefig("outputComparison/"+year+"/2D/{model}/{dnnName}_purity_{sample}_{nr}.pdf".format(model=modelPath.split("/")[-1], dnnName=dnnName, sample=sampName, nr=str(modelNr)))
            
            # Plotting Stability
            
            fig, ax = plt.subplots(1,1)
            #  ~#fig.suptitle("{}, {} sample\n, {}".format(dnnName, sampName, title), fontsize=12, ha="left", x=0.1, y=0.99)
            mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both/histo2D_Gen, vmin=0., vmax=vmax)
            for i,phiarr in enumerate(histo2D_Both/histo2D_Gen):
                for j,vali in enumerate(phiarr):
                    if optBins: ax.text(metBinsC[j]-15, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=10, color="red", fontweight="bold")
                    else: ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=14, color="red", fontweight="bold")
                    # Plotting uncertainties:
                    #  ~tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Gen[i,j])
                    #  ~ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
                    #  ~ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=7, color="red")
            cbar = fig.colorbar(mesh1, ax=ax)
            cbar.set_label("stability")
            ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
            ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
            makeCMStitle(year)
            plt.savefig("outputComparison/"+year+"/2D/{model}/{dnnName}_stability_{sample}_{nr}.pdf".format(model=modelPath.split("/")[-1], dnnName=dnnName, sample=sampName, nr=str(modelNr)))
            
            
            # Plotting Response Matrix
            
            Xbins, Ybins = np.meshgrid(np.arange(nB+1),np.arange(nB+1))
            binsC = (np.arange(nB+1)[1:]+np.arange(nB+1)[:-1])*0.5
            
            fig, ax = plt.subplots(figsize=(9,5))
            mesh1 = ax.pcolormesh(Xbins, Ybins, histoResponse, vmin=0., vmax=vmax)
            for i,row in enumerate(histoResponse):
                for j,vali in enumerate(row):
                    ax.text(binsC[j]-0.4, binsC[i]-0.25, "{:.2f}".format(vali)[1:], fontsize=9, color="red")
                    # Plotting uncertainties:
                    # tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Gen[i,j])
                    # ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
            cbar = fig.colorbar(mesh1, ax=ax)
            cbar.set_label("line normalized distribution")
            ax.set_ylabel(r"reco bin number")
            ax.set_xlabel(r"gen bin number")
            if optBins: 
                ax.axhline(7., color="lime", linestyle="dashed")
                ax.axhline(14., color="lime", linestyle="dashed")
                ax.axvline(7., color="lime", linestyle="dashed")
                ax.axvline(14., color="lime", linestyle="dashed")
                ax.set_xticks([0.5,3.5,6.5,9.5,12.5,15.5,18.5])
                ax.set_xticklabels(["1","4","7","10","13","16","19"])
                ax.set_yticks([0.5,3.5,6.5,9.5,12.5,15.5,18.5])
                ax.set_yticklabels(["1","4","7","10","13","16","19"])
            else:
                ax.axhline(6., color="lime", linestyle="dashed")
                ax.axhline(12., color="lime", linestyle="dashed")
                ax.axvline(6., color="lime", linestyle="dashed")
                ax.axvline(12., color="lime", linestyle="dashed")
                ax.set_xticks([0.5,3.5,6.5,9.5,12.5,15.5])
                ax.set_xticklabels(["1","4","7","10","13","16"])
                ax.set_yticks([0.5,3.5,6.5,9.5,12.5,15.5])
                ax.set_yticklabels(["1","4","7","10","13","16"])
            makeCMStitle(year)
            plt.tight_layout()
            plt.savefig("outputComparison/"+year+"/2D/{model}/{dnnName}_Response_{sample}_{nr}.pdf".format(model=modelPath.split("/")[-1], dnnName=dnnName, sample=sampName, nr=str(modelNr)))

    
#############################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nominal', type=bool, help="Use nominal TTbar_diLepton sample instead of TTbar_amcatnlo")
    parser.add_argument('--year', type=str, help="Year, choose between 2016_preVFP, 2016_postVFP, 2017, 2018 (default)")
    parser.add_argument('--version', type=str, help="treeVersion, such as v07 (default)")
    parser.add_argument('--mode', type=int, help="Runninge mode, 1 (default) for plotting performance, 2 for plotting purity, 3 for printing target values")
    args = parser.parse_args()

    year = "2018"
    version = "v07"
    ttbarDilepNominal = False
    mode = 1
    
    if args.year: year = args.year
    if args.version: version = args.version
    if args.Nominal: ttbarDilepNominal = args.Nominal
    if args.mode: mode = args.mode

    dataPath = "/net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/minTrees/100.0/Nominal/".format(year=year, version=version)
    
    if ttbarDilepNominal:
        sampleNameRoot = "TTbar_diLepton"
        dataPath += sampleNameRoot+"_merged.root"
    else:
        sampleNameRoot = "TTbar_amcatnlo"
        dataPath += sampleNameRoot+"_merged.root"
    
    
    # Define Input Variables
    #  ~inputVars = ["PuppiMET_xy*cos(PuppiMET_xy_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)","METunc_Puppi","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","HT*cos(HT_phi)","HT*sin(HT_phi)","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt*cos(Lep1_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_eta","Lep1_E","Lep2_pt*cos(Lep2_phi)","Lep2_pt*sin(Lep2_phi)","Lep2_eta","Lep2_E","Jet1_pt*cos(Jet1_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_eta","Jet1_E","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","CaloMET*cos(CaloMET_phi)","CaloMET*sin(CaloMET_phi)","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj"]   # All input variables
    inputVars = ["PuppiMET_xy*sin(PuppiMET_xy_phi)", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "MET_xy*sin(MET_xy_phi)", "MET_xy*cos(MET_xy_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "mjj", "Jet1_E", "HT", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)"] # Finalized set of input variables
    

    # Defining a model ID (modelNR) to be able to differentiate the plot names from other iterations, the Model Path to be plotted and the Network structure
    modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220527-1204genMETweighted"
    netStruct = {"alph": 0.11561571686757542, "batch": 6.848504823076402, "dout": 0.1396968795456421, "lamb": -7.595457622529098, "lr": -8.975982634440493, "nLayer": 4.887771333464286, "nodeFac": 0.33526528228286} #genMETrew, final inputs
    nInputs=int(len(inputVars))
    modelNr = 102
    
    
    alph, batch, dout, lamb, lr, nLayer, nodeFac = netStruct["alph"], netStruct["batch"], netStruct["dout"], netStruct["lamb"], netStruct["lr"], netStruct["nLayer"], netStruct["nodeFac"] 
    
    # Printing the DNN structure
    nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]
    modelTitle = r"learnrate={:.2e};dropout={:.2};$\lambda$={:.2e};batchsize={:.2e};".format(np.exp(lr),dout,np.exp(lamb),int(np.round(np.exp(batch))))+"\n"+r"n_layer={};nodefactor={};$\alpha$={:.2};n_inputs={};genMETreweighting".format(int(np.round(nLayer)),nodeFacs[int(np.round(nodeFac))],alph,nInputs)
    print(modelTitle,"\n")
    
    if mode==2:
        # Plotting the Purity, Stability and Response Matrix for the training sample, splitting 99% of the data into training sample for reasonable statistics in high MET bins
        plot_Purity(year,dataPath,inputVars[:nInputs],modelPath,sampleNameRoot,"diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=True,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)
    elif mode==3:
        # Printing some target values of the chosen model, such as the loss or purity in last bin; Not yet updated to incorporate SF_weights
        print_targets(dataPath,inputVars[:nInputs],modelPath,sampleNameRoot,"diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=True,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)
    else:
        # Creating Plots of the Network performance, using the same training, validation and test sample as in the training process
        plot_Output(year,dataPath,inputVars[:nInputs],modelPath,sampleNameRoot,"diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)
    
